import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from pytorch_utils.modules import MLP
from kbp_2015_utils.constants import EVENT_TYPES, REALIS_VALS


class Controller(BaseController):
    def __init__(self, **kwargs):
        super(Controller, self).__init__(**kwargs)

        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def get_gold_mentions(self, clusters, num_words, flat_cand_mask):
        if self.training and self.label_smoothing_wt > 0.0:
            gold_ments = self.label_smoothing_wt * torch.ones(num_words, self.max_span_width).cuda()
        else:
            gold_ments = torch.zeros(num_words, self.max_span_width).cuda()

        for cluster in clusters:
            for mention in cluster:
                span_start, span_end, mention_info = mention

                span_width = span_end - span_start + 1
                if span_width <= self.max_span_width:
                    span_width_idx = span_width - 1
                    gold_ments[span_start, span_width_idx] = (1.0 - self.label_smoothing_wt if self.training else 1.0)

        return gold_ments.reshape(-1)[flat_cand_mask].float()

    def forward(self, example, teacher_forcing=False, final_eval=False, debug=False):
        """
        Encode a batch of excerpts.
        """
        output = self.doc_encoder(example)
        encoded_doc = output[0]
        num_words = encoded_doc.shape[0]
        assert(num_words == sum([len(sentence) for sentence in example["sentences"]]))

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)
        span_embs = self.get_span_embeddings(example["doc_type"], encoded_doc, filt_cand_starts, filt_cand_ends)

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)
        pred_mention_probs = {}
        loss = {}

        mention_logits = torch.squeeze(self.other.mention_mlp(span_embs), dim=-1)
        ment_width_scores = self.get_mention_width_scores(
            filt_cand_starts, filt_cand_ends, example["subtoken_map"])
        mention_logits += torch.squeeze(ment_width_scores, dim=-1)

        if self.training:
            mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
            total_weight = filt_cand_starts.shape[0]

            loss['ment_loss'] = mention_loss / total_weight
        else:
            pred_mention_probs = torch.sigmoid(mention_logits).detach()

        if self.training:
            return loss
        else:
            return pred_mention_probs, filt_gold_mentions, flat_cand_mask
