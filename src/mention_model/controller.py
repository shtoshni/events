import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController


class Controller(BaseController):
    def __init__(self,  **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, example, teacher_forcing=False, final_eval=False, debug=False):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
        num_words = encoded_doc.shape[0]
        assert(num_words == sum([len(sentence) for sentence in example["sentences"]]))

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)
        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)
        pred_mention_probs = {}
        loss = {}

        for category in ["event_subtype", "event_type", "realis"]:
            mention_logits = torch.squeeze(self.other.mention_mlp[category](span_embs), dim=-1)
            if category == "event_subtype" or category == "event_type":
                ment_width_scores = self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)
                mention_logits += torch.unsqueeze(ment_width_scores, dim=-1)

            if self.training:
                mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions[category])
                total_weight = filt_cand_starts.shape[0]

                loss[category] = mention_loss / total_weight
            else:
                pred_mention_probs[category] = torch.sigmoid(mention_logits).detach()

        if self.training:
            return loss
        else:
            return pred_mention_probs["event_subtype"], filt_gold_mentions["event_subtype"]
