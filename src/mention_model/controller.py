import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from pytorch_utils.modules import MLP
from red_utils.constants import IDX_TO_ELEM_TYPE, DOC_TYPE_TO_IDX
from document_encoder import IndependentDocEncoder, OverlapDocEncoder
from pytorch_memlab import profile, set_target_gpu
from red_utils.utils import get_doc_type

ELEM_TO_TOP_SPAN_RATIO = {'ENTITY': 0.25, 'EVENT': 0.2}


class Controller(BaseController):
    def __init__(self, mlp_size=1024, mlp_depth=1, max_span_width=30, top_span_ratio=0.4,
                 ment_emb='endpoint', dropout_rate=0.5, doc_enc='independent',
                 **kwargs):
        super(Controller, self).__init__()

        self.max_span_width = max_span_width

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(**kwargs)
        else:
            self.doc_encoder = OverlapDocEncoder(**kwargs)

        self.hsize = self.doc_encoder.hsize
        self.emb_size = 20
        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        self.max_span_width = max_span_width
        self.mlp_size = mlp_size
        self.mlp_depth = mlp_depth
        self.top_span_ratio = top_span_ratio

        self.other = nn.Module()

        self.other.span_width_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.other.span_width_prior_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.other.doc_type_emb = nn.Embedding(3, self.emb_size)

        self.other.mention_mlp = nn.ModuleDict()
        for elem_type in IDX_TO_ELEM_TYPE[:2]:
            self.other.mention_mlp[elem_type] = MLP(
                input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + 2 * self.emb_size,
                hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                bias=True, drop_module=self.drop_module)
        self.other.span_width_mlp = MLP(
            input_size=20, hidden_size=self.mlp_size,
            output_size=1, num_hidden_layers=1, bias=True,
            drop_module=self.drop_module)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def get_mention_width_scores(self, cand_starts, cand_ends):
        span_width_idx = cand_ends - cand_starts
        span_width_embs = self.other.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.other.span_width_mlp(span_width_embs), dim=-1)

        return width_scores

    def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends, doc_type):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]

        # Add span width embeddings
        span_width_indices = ment_ends - ment_starts
        span_width_embs = self.other.span_width_embeddings(span_width_indices)
        span_emb_list.append(span_width_embs)

        doc_type_idx = DOC_TYPE_TO_IDX[doc_type]
        doc_type_emb = self.other.doc_type_emb(torch.tensor(doc_type_idx).long().cuda())
        doc_type_emb = torch.unsqueeze(doc_type_emb, dim=0)
        doc_type_emb = doc_type_emb.repeat(span_width_embs.shape[0], 1)
        span_emb_list.append(doc_type_emb)

        if self.ment_emb == 'attn':
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words), 0).repeat(num_c, 1).cuda()  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]
            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]

            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # C x H
            span_emb_list.append(attention_term)

        span_embs = torch.cat(span_emb_list, dim=-1)
        return span_embs

    def get_gold_mentions(self, clusters, num_words, flat_cand_mask, ment_type):
        gold_ments = torch.zeros(num_words, self.max_span_width).cuda()
        for cluster in clusters:
            for mention in cluster:
                span_start, span_end, span_type = mention
                if span_type == ment_type:
                    span_width = span_end - span_start + 1
                    if span_width <= self.max_span_width:
                        span_width_idx = span_width - 1
                        gold_ments[span_start, span_width_idx] = 1

        filt_gold_ments = gold_ments.reshape(-1)[flat_cand_mask].float()
        # assert(torch.sum(gold_ments) == torch.sum(filt_gold_ments))  # Filtering shouldn't remove gold mentions
        return filt_gold_ments

    def get_candidate_endpoints(self, encoded_doc, example):
        num_words = encoded_doc.shape[0]

        sent_map = torch.tensor(example["sentence_map"]).cuda()
        # num_words x max_span_width
        cand_starts = torch.unsqueeze(torch.arange(num_words), dim=1).repeat(1, self.max_span_width).cuda()
        cand_ends = cand_starts + torch.unsqueeze(torch.arange(self.max_span_width), dim=0).cuda()

        cand_start_sent_indices = sent_map[cand_starts]
        # Avoid getting sentence indices for cand_ends >= num_words
        corr_cand_ends = torch.min(cand_ends, torch.ones_like(cand_ends).cuda() * (num_words - 1))
        cand_end_sent_indices = sent_map[corr_cand_ends]

        # End before document ends & Same sentence
        constraint1 = (cand_ends < num_words)
        # Removing this constraint because RED doesn't have sentence segmentation
        constraint2 = (cand_start_sent_indices == cand_end_sent_indices)
        cand_mask = constraint1 & constraint2
        flat_cand_mask = cand_mask.reshape(-1)

        # Filter and flatten the candidate end points
        filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        return filt_cand_starts, filt_cand_ends, flat_cand_mask

    # @profile
    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends,
                                             get_doc_type(example))
        del encoded_doc
        ment_width_scores = self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        loss = {}
        pred_mention_probs = {}
        filt_gold_mentions_dict = {}
        recall = {}
        for ment_idx, ment_type in enumerate(IDX_TO_ELEM_TYPE[:2]):
            if ment_type == 'BOTH':
                continue
            mention_logits = torch.squeeze(self.other.mention_mlp[ment_type](span_embs), dim=-1)
            mention_logits += ment_width_scores

            filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask, ment_idx)
            filt_gold_mentions_dict[ment_type] = filt_gold_mentions

            # if not self.training and ment_type == 'ENTITY':
            #     print(example["doc_key"], torch.sum(filt_gold_mentions_dict[ment_type]).item())

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

                if not self.training or teacher_forcing:
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
                                        pass

        if self.training:
            return loss
        else:
            return pred_mention_probs, filt_gold_mentions_dict, filt_cand_starts, filt_cand_ends, recall
