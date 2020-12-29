import torch
import torch.nn as nn

from collections import Counter, OrderedDict
from document_encoder.independent import IndependentDocEncoder
from pytorch_utils.modules import MLP
from kbp_2015_utils.constants import EVENT_SUBTYPES, EVENT_TYPES, REALIS_VALS
from auto_memory_model.constants import SPANS_TO_LEN_RATIO
from data_utils.utils import get_clusters


class BaseController(nn.Module):
    def __init__(self,
                 dropout_rate=0.5, max_span_width=20,
                 ment_emb='endpoint', ment_ordering='ment_type',
                 mlp_size=1000, emb_size=20,
                 sample_invalid=1.0, label_smoothing_wt=0.0,
                 dataset='kbp_2015',  focus_group='both',
                 **kwargs):
        super(BaseController, self).__init__()
        self.dataset = dataset

        self.ment_ordering = ment_ordering
        self.max_span_width = max_span_width
        self.sample_invalid = sample_invalid
        self.label_smoothing_wt = label_smoothing_wt

        self.doc_encoder = IndependentDocEncoder(**kwargs)

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

        for category, category_vals in zip(["event_subtype"], [EVENT_SUBTYPES]):
            self.other.mention_mlp[category] = MLP(
                input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + 2 * self.emb_size,
                hidden_size=self.mlp_size, output_size=len(category_vals), num_hidden_layers=1,
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

    def get_span_embeddings(self, doc_type, encoded_doc, ment_starts, ment_ends):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]

        # Add span width embeddings
        span_width_indices = ment_ends - ment_starts
        span_width_embs = self.other.span_width_embeddings(span_width_indices)
        span_emb_list.append(span_width_embs)

        doc_type_emb = self.other.doc_type_emb(torch.tensor(doc_type).long().cuda())
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
        # Removing this constraint because sentence boundaries based on simple heuristics such as "\n"
        # don't work for KBP
        constraint2 = (cand_start_sent_indices == cand_end_sent_indices)
        cand_mask = constraint1 & constraint2
        # cand_mask = constraint1
        flat_cand_mask = cand_mask.reshape(-1)

        # Filter and flatten the candidate end points
        filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        return filt_cand_starts, filt_cand_ends, flat_cand_mask

    def get_pred_mentions(self, example, encoded_doc, category="event_subtype"):
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(encoded_doc, example)

        span_embs = self.get_span_embeddings(example["doc_type"], encoded_doc, filt_cand_starts, filt_cand_ends)
        ment_width_scores = self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        mention_logits = torch.squeeze(self.other.mention_mlp[category](span_embs), dim=-1)
        mention_logits += torch.unsqueeze(ment_width_scores, dim=-1)

        k = int(SPANS_TO_LEN_RATIO[self.dataset] * num_words)
        num_ment_type = mention_logits.shape[1]  # Number of event subtypes
        mention_logits = mention_logits.reshape(-1)
        reshaped_topk_indices = torch.topk(mention_logits, k)[1]

        topk_indices = reshaped_topk_indices // num_ment_type
        topk_ment_type = reshaped_topk_indices % num_ment_type

        topk_indices_mask = torch.zeros_like(mention_logits).cuda()
        topk_indices_mask[topk_indices] = 1

        topk_starts = filt_cand_starts[topk_indices]
        topk_ends = filt_cand_ends[topk_indices]
        topk_scores = mention_logits[reshaped_topk_indices]

        # Sort the mentions by (start) and tiebreak with (end)
        sort_scores = topk_starts + 1e-5 * topk_ends
        _, sorted_indices = torch.sort(sort_scores, dim=0)

        if self.ment_ordering == 'document':
            # Order mentions by their order in document
            sort_scores = topk_starts + 1e-5 * topk_ends + 1e-5 * topk_ment_type
            _, sorted_indices = torch.sort(sort_scores, dim=0)

            return (topk_starts[sorted_indices], topk_ends[sorted_indices],
                    topk_scores[sorted_indices], topk_ment_type[sorted_indices])
        else:
            # Else mentions remain ordered by their type
            return topk_starts, topk_ends, topk_scores, topk_ment_type

    def get_mention_embs_and_actions(self, example):
        encoded_doc = self.doc_encoder(example)
        pred_starts, pred_ends, pred_scores, pred_ment_type = self.get_pred_mentions(example, encoded_doc)

        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist(), pred_ment_type.tolist()))
        mention_embs = self.get_span_embeddings(example["doc_type"], encoded_doc, pred_starts, pred_ends)
        mention_emb_list = torch.unbind(mention_embs, dim=0)
        pred_scores_list = torch.unbind(torch.unsqueeze(pred_scores, dim=1))

        if self.dataset == "kbp_2015":
            clusters = get_clusters(example["clusters"], key="subtype_val")
        else:
            clusters = example["clusters"]
        gt_actions = self.get_actions(pred_mentions, clusters)
        return pred_mentions, gt_actions, mention_emb_list, pred_scores_list

    def forward(self, example, teacher_forcing=False):
        pass
