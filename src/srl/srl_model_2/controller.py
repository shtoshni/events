import torch
import torch.nn as nn

from pytorch_utils.modules import MLP
from srl.constants import LABELS
from srl.srl_model_2.encoder import IndependentDocEncoder


class Controller(nn.Module):
    def __init__(self, dropout_rate=0.5, ment_emb='endpoint',
                 mlp_size=1000, emb_size=20, dataset='conll09',
                 **kwargs):
        super(Controller, self).__init__()
        self.dataset = dataset
        self.doc_encoder = IndependentDocEncoder(**kwargs)

        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.drop_module = nn.Dropout(p=dropout_rate)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        self.hsize = self.doc_encoder.hsize

        self.span_emb_size = self.ment_emb_to_size_factor[self.ment_emb] * self.hsize
        self.other = nn.Module()
        self.other.arg_pred_biaffine = nn.Bilinear(self.span_emb_size, self.span_emb_size, len(LABELS), bias=True)

        self.other.arg_mlp = MLP(
            input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
            hidden_size=self.mlp_size, output_size=len(LABELS), num_hidden_layers=1,
            bias=True, drop_module=self.drop_module)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    @staticmethod
    def get_predicate_embedding(example, encoded_doc):
        start_idx, end_idx = example["start_idx"], example["end_idx"]
        start_pred_embs = encoded_doc[start_idx[example["predicate"]]]
        end_pred_embs = encoded_doc[end_idx[example["predicate"]]]

        return torch.cat([start_pred_embs, end_pred_embs], dim=0)

    @staticmethod
    def get_token_embedding(example, encoded_doc, num_pad_tokens=0):
        start_embs = encoded_doc[torch.tensor(example["start_idx"]).cuda()]
        end_embs = encoded_doc[torch.tensor(example["end_idx"]).cuda()]
        token_embs = torch.cat([start_embs, end_embs], dim=1)
        if num_pad_tokens:
            padding = torch.zeros(num_pad_tokens, token_embs.shape[1]).cuda()
            token_embs = torch.cat([token_embs, padding], dim=0)
        return token_embs

    @staticmethod
    def get_seq_labels(example, num_pad_tokens=0):
        labels = [0] * len(example["start_idx"] + [-100] * num_pad_tokens)
        for arg_info in example["args"]:
            token_idx, label_idx = arg_info[:2]
            labels[token_idx] = label_idx
        return torch.tensor(labels).cuda()

    def forward(self, example):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)

        num_words = encoded_doc.shape[0]
        assert(num_words == sum([len(sentence) for sentence in example["sentences"]]))

        pred_embs = self.get_predicate_embedding(example, encoded_doc)
        token_embs = self.get_token_embedding(example, encoded_doc)
        seq_labels = self.get_seq_labels(example)

        num_tokens = token_embs.shape[0]
        pred_embs = torch.unsqueeze(pred_embs, dim=0).repeat(num_tokens, 1)

        unary_score = self.other.arg_mlp(token_embs)
        pairwise_score = self.other.arg_pred_biaffine(pred_embs, token_embs)
        score = unary_score + pairwise_score

        loss = self.loss_fn(score, seq_labels)

        if self.training:
            return loss
        else:
            # L x P
            token_pred_list = torch.argmax(score, dim=1).tolist()
            argument_list = []

            for token_idx, arg_pred in enumerate(token_pred_list):
                if arg_pred != 0:
                    argument_list.append((token_idx, arg_pred))

            return argument_list

    def forward2(self, example_list):
        max_tokens = 0
        for example in example_list:
            num_tokens = len(example["subtoken_map"])
            if max_tokens < num_tokens:
                max_tokens = num_tokens

        encoded_doc_list = []
        pred_embedding_list = []
        token_embedding_list = []
        seq_label_list = []
        for example in example_list:
            num_tokens = len(example["subtoken_map"])
            num_pad_tokens = max_tokens - num_tokens
            encoded_doc = self.doc_encoder(example)
            encoded_doc_list.append(encoded_doc)
            pred_embedding_list.append(self.get_predicate_embedding(example, encoded_doc))
            token_embedding_list.append(self.get_token_embedding(example, encoded_doc, num_pad_tokens=num_pad_tokens))
            seq_label_list.append(self.get_seq_labels(example))
