import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from red_utils.constants import IDX_TO_ELEM_TYPE
from red_utils.utils import get_doc_type

ELEM_TO_TOP_SPAN_RATIO = {'ENTITY': 0.3, 'EVENT': 0.2}


class Controller(BaseController):
    def __init__(self,  **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, example, teacher_forcing=False, final_eval=False):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)
        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends, get_doc_type(example))

        ment_width_scores = self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        loss = {}
        pred_mention_probs = {}
        filt_gold_mentions_dict = {}
        recall = {}
        for ment_idx, ment_type in enumerate(IDX_TO_ELEM_TYPE):
            mention_logits = torch.squeeze(self.other.mention_mlp[ment_type](span_embs), dim=-1)
            mention_logits += ment_width_scores

            filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask, ment_idx)
            filt_gold_mentions_dict[ment_type] = filt_gold_mentions

            if self.training:
                mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
                total_weight = filt_cand_starts.shape[0]

                loss[ment_type] = mention_loss / total_weight
            else:
                pred_mention_probs[ment_type] = torch.sigmoid(mention_logits).detach()
                # Calculate Recall
                k = int(ELEM_TO_TOP_SPAN_RATIO[ment_type] * num_words)
                topk_indices = torch.topk(mention_logits, k)[1]
                topk_indices_mask = torch.zeros_like(mention_logits).cuda()
                topk_indices_mask[topk_indices] = 1
                recall[ment_type] = torch.sum(filt_gold_mentions * topk_indices_mask).item()

                if final_eval:
                    diff_vec = filt_gold_mentions - filt_gold_mentions * topk_indices_mask
                    if torch.sum(diff_vec):
                        print(example["doc_key"], ment_type, torch.sum(diff_vec))
                        doc = []
                        for sentence in example["sentences"]:
                            doc.extend(sentence)

                        top_k_starts = filt_cand_starts[topk_indices]
                        top_k_ends = filt_cand_ends[topk_indices]
                        top_k_pairs = torch.stack([top_k_starts, top_k_ends], dim=1).tolist()

                        top_k_set = set()
                        for pair in top_k_pairs:
                            top_k_set.add(tuple(pair))

                        # print(sorted(top_k_pairs))
                        for cluster in example["clusters"]:
                            for span_start, span_end, span_type in cluster:
                                if span_type == ment_idx:
                                    span_tuple = (span_start, span_end)
                                    if span_tuple not in top_k_set:
                                        print(span_tuple, doc[span_start: span_end + 1])

        if self.training:
            return loss
        else:
            return pred_mention_probs, filt_gold_mentions_dict, filt_cand_starts, filt_cand_ends, recall
