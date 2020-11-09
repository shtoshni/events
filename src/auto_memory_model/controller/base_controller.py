import torch
import torch.nn as nn

from collections import Counter, OrderedDict
from document_encoder.independent import IndependentDocEncoder
from document_encoder.overlap import OverlapDocEncoder
from pytorch_utils.modules import MLP
from red_utils.constants import ELEM_TYPE_TO_IDX, IDX_TO_ELEM_TYPE, DOC_TYPE_TO_IDX
from red_utils.utils import get_doc_type

ELEM_TO_TOP_SPAN_RATIO = {'ENTITY': 0.3, 'EVENT': 0.2}


class BaseController(nn.Module):
    def __init__(self,
                 dropout_rate=0.5, max_span_width=20,
                 ment_emb='endpoint', doc_enc='independent', ment_ordering='ment_type',
                 mlp_size=1000, emb_size=20,
                 sample_invalid=1.0, label_smoothing_wt=0.0,
                 dataset='red',  focus_group='both',
                 **kwargs):
        super(BaseController, self).__init__()
        self.dataset = dataset

        self.ment_ordering = ment_ordering
        self.max_span_width = max_span_width
        self.sample_invalid = sample_invalid
        self.label_smoothing_wt = label_smoothing_wt

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(**kwargs)
        else:
            self.doc_encoder = OverlapDocEncoder(**kwargs)

        self.hsize = self.doc_encoder.hsize
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.drop_module = nn.Dropout(p=dropout_rate)
        self.ment_emb = ment_emb
        self.focus_group = focus_group
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        self.other = nn.Module()
        self.other.span_width_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.other.span_width_prior_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.other.doc_type_emb = nn.Embedding(3, self.emb_size)

        self.other.mention_mlp = nn.ModuleDict()
        for elem_type in IDX_TO_ELEM_TYPE:
            self.other.mention_mlp[elem_type] = MLP(
                input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + 2 * self.emb_size,
                hidden_size=self.mlp_size, output_size=1, num_hidden_layers=1,
                bias=True, drop_module=self.drop_module)
        self.other.span_width_mlp = MLP(
            input_size=20, hidden_size=self.mlp_size,
            output_size=1, num_hidden_layers=1, bias=True,
            drop_module=self.drop_module)

        self.memory_net = None
        self.loss_fn = {}

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

    def get_pred_mentions(self, example, encoded_doc):
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends, get_doc_type(example))
        ment_width_scores = self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        if self.focus_group == 'joint':
            elem_types = IDX_TO_ELEM_TYPE
        else:
            elem_types = [self.focus_group]

        all_topk_starts = None
        all_topk_ends = None
        all_topk_scores = None
        all_ment_type = None

        for ment_type in elem_types:
            ment_idx = ELEM_TYPE_TO_IDX[ment_type]
            mention_logits = torch.squeeze(self.other.mention_mlp[ment_type](span_embs), dim=-1)
            mention_logits += ment_width_scores

            k = int(ELEM_TO_TOP_SPAN_RATIO[ment_type] * num_words)
            topk_indices = torch.topk(mention_logits, k)[1]

            topk_indices_mask = torch.zeros_like(mention_logits).cuda()
            topk_indices_mask[topk_indices] = 1

            topk_starts = filt_cand_starts[topk_indices]
            topk_ends = filt_cand_ends[topk_indices]
            topk_scores = mention_logits[topk_indices]

            # Sort the mentions by (start) and tiebreak with (end)
            sort_scores = topk_starts + 1e-5 * topk_ends
            _, sorted_indices = torch.sort(sort_scores, dim=0)

            if all_topk_starts is None:
                all_topk_starts = topk_starts[sorted_indices]
                all_topk_ends = topk_ends[sorted_indices]
                all_topk_scores = topk_scores[sorted_indices]
                all_ment_type = torch.tensor([ment_idx] * sorted_indices.shape[0]).cuda()
            else:
                all_topk_starts = torch.cat([all_topk_starts, topk_starts[sorted_indices]], dim=0)
                all_topk_ends = torch.cat([all_topk_ends, topk_ends[sorted_indices]], dim=0)
                all_topk_scores = torch.cat([all_topk_scores, topk_scores[sorted_indices]], dim=0)
                all_ment_type = torch.cat([all_ment_type,
                                           torch.tensor([ment_idx] * sorted_indices.shape[0]).cuda()], dim=0)

        if self.ment_ordering == 'document':
            # Order mentions by their order in document
            sort_scores = all_topk_starts + 1e-5 * all_topk_ends + 1e-5 * all_ment_type
            _, sorted_indices = torch.sort(sort_scores, dim=0)

            return (all_topk_starts[sorted_indices], all_topk_ends[sorted_indices],
                    all_topk_scores[sorted_indices], all_ment_type[sorted_indices])
        else:
            # Else mentions remain ordered by their type
            return (all_topk_starts, all_topk_ends, all_topk_scores, all_ment_type)

    def get_mention_embs_and_actions(self, example):
        encoded_doc = self.doc_encoder(example)
        pred_starts, pred_ends, pred_scores, pred_ment_type = self.get_pred_mentions(example, encoded_doc)

        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist(), pred_ment_type.tolist()))
        mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends, get_doc_type(example))
        mention_emb_list = torch.unbind(mention_embs, dim=0)
        pred_scores_list = torch.unbind(torch.unsqueeze(pred_scores, dim=1))

        gt_actions = self.get_actions(pred_mentions, example["clusters"])
        return pred_mentions, gt_actions, mention_emb_list, pred_scores_list

    def forward(self, example, teacher_forcing=False):
        pass
