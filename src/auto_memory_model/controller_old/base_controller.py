import random
import torch
import torch.nn as nn

from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder import IndependentDocEncoder, OverlapDocEncoder
from auto_memory_model.utils import get_ordered_mentions
from red_utils.constants import ELEM_TYPE_TO_IDX


class BaseController(nn.Module):
    def __init__(self,
                 dropout_rate=0.5, max_span_width=20, focus_group='both',
                 ment_emb='endpoint', doc_enc='independent',
                 sample_singletons=1.0, label_smoothing_wt=0.1, label_smoothing_other=False,
                 **kwargs):
        super(BaseController, self).__init__()
        self.max_span_width = max_span_width

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(**kwargs)
        else:
            self.doc_encoder = OverlapDocEncoder(**kwargs)

        self.hsize = self.doc_encoder.hsize
        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.ment_emb = ment_emb
        self.focus_group = focus_group
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        self.sample_singletons = sample_singletons
        self.label_smoothing_wt = label_smoothing_wt
        self.label_smoothing_other = label_smoothing_other

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        self.memory_net = None
        self.loss_fn = {}

    def get_mention_embeddings(self, encoded_doc, ment_starts, ment_ends):
        ment_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]

        if self.ment_emb == 'endpoint':
            return torch.cat(ment_emb_list, dim=-1)
        else:
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words), 0).repeat(num_c, 1).cuda()  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]
            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]
            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # K x H

            ment_emb_list.append(attention_term)
            return torch.cat(ment_emb_list, dim=1)

    def get_document_enc(self, example):
        if self.doc_enc == 'independent':
            encoded_output = self.doc_enc(example)
        else:
            # Overlap
            encoded_output = None

        return encoded_output

    def get_mention_embs_and_actions(self, example):
        encoded_output = self.doc_encoder(example)

        clusters = example["clusters"]
        if self.sample_singletons < 1.0 and self.training:
            clusters = [cluster for cluster in clusters
                        if (len(cluster) > 1) or (random.random() <= self.sample_singletons)]

        gt_mentions = get_ordered_mentions(clusters)
        pred_mentions = gt_mentions
        # print(self.focus_group)
        if self.focus_group == 'entity':
            pred_mentions = [mention for mention in pred_mentions if mention[2] == ELEM_TYPE_TO_IDX['ENTITY']]
        elif self.focus_group == 'event':
            pred_mentions = [mention for mention in pred_mentions if mention[2] == ELEM_TYPE_TO_IDX['EVENT']]

        if len(pred_mentions):
            gt_actions = self.get_actions(pred_mentions, clusters)

            cand_starts, cand_ends, ent_type = zip(*pred_mentions)
            mention_embs = self.get_mention_embeddings(
                encoded_output, torch.tensor(cand_starts).cuda(), torch.tensor(cand_ends).cuda())
            mention_emb_list = torch.unbind(mention_embs, dim=0)
            return gt_mentions, pred_mentions, gt_actions, mention_emb_list
        else:
            return [], [], [], []

    def forward(self, example, teacher_forcing=False):
        pass