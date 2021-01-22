import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from pytorch_utils.modules import MLP
from kbp_2015_utils.constants import EVENT_TYPES, REALIS_VALS


class Controller(BaseController):
    def __init__(self,  **kwargs):
        super(Controller, self).__init__(**kwargs)
        for category, category_vals in zip(["event_type", "realis"], [EVENT_TYPES, REALIS_VALS]):
            self.other.mention_mlp[category] = MLP(
                input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + 2 * self.emb_size,
                hidden_size=self.mlp_size, output_size=len(category_vals), num_hidden_layers=1,
                bias=True, drop_module=self.drop_module)

        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, example, teacher_forcing=False, final_eval=False, debug=False):
        """
        Encode a batch of excerpts.
        """
        output = self.doc_encoder(example)
        encoded_doc = output[0]
        num_words = encoded_doc.shape[0]
        assert(num_words == sum([len(sentence) for sentence in example["sentences"]]))

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)
        span_embs = self.get_span_embeddings(example["doc_type"], encoded_doc, filt_cand_starts,
                                             filt_cand_ends, example["subtoken_map"])

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)
        pred_mention_probs = {}
        loss = {}

        for category in ["event_subtype", "event_type", "realis"]:
            mention_logits = torch.squeeze(self.other.mention_mlp[category](span_embs), dim=-1)
            if category == "event_subtype" or category == "event_type":
                ment_width_scores = self.get_mention_width_scores(
                    filt_cand_starts, filt_cand_ends, example["subtoken_map"])
                mention_logits += torch.unsqueeze(ment_width_scores, dim=-1)

            if self.training:
                mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions[category])
                total_weight = filt_cand_starts.shape[0]

                loss[category] = mention_loss / total_weight
                if self.doc_encoder.use_srl:
                    loss['srl'] = output[1]
            else:
                pred_mention_probs[category] = torch.sigmoid(mention_logits).detach()

        if self.training:
            return loss
        else:
            return pred_mention_probs["event_subtype"], pred_mention_probs["realis"], \
                   filt_gold_mentions["event_subtype"], flat_cand_mask
