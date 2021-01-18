import torch
import torch.nn as nn

from pytorch_utils.modules import MLP
from srl.constants import LABELS
from document_encoder.independent import IndependentDocEncoder


class Controller(nn.Module):
    def __init__(self, dropout_rate=0.5, ment_emb='endpoint',
                 mlp_size=1000, emb_size=20, dataset='conll09',
                 **kwargs):
        super(Controller, self).__init__()
        self.dataset = dataset
        self.doc_encoder = IndependentDocEncoder(**kwargs)

        self.hsize = self.doc_encoder.hsize
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.drop_module = nn.Dropout(p=dropout_rate)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        self.other = nn.Module()

        self.other.arg_pred_mlp = MLP(
            input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize * 2,
            hidden_size=self.mlp_size, output_size=len(LABELS), num_hidden_layers=1,
            bias=True, drop_module=self.drop_module)

        self.other.arg_mlp = MLP(
            input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
            hidden_size=self.mlp_size, output_size=len(LABELS), num_hidden_layers=1,
            bias=True, drop_module=self.drop_module)

        self.loss_fn = nn.CrossEntropyLoss()

    def get_predicate_embedding(self, example, encoded_doc):
        start_idx, end_idx = example["start_idx"], example["end_idx"]
        start_pred_embs = encoded_doc[torch.tensor([start_idx[t_idx] for t_idx in example["predicate"]]).cuda()]
        end_pred_embs = encoded_doc[torch.tensor([end_idx[t_idx] for t_idx in example["predicate"]]).cuda()]

        return torch.cat([start_pred_embs, end_pred_embs], dim=1)

    def get_token_embedding(self, example, encoded_doc):
        start_embs = encoded_doc[torch.tensor(example["start_idx"]).cuda()]
        end_embs = encoded_doc[torch.tensor(example["end_idx"]).cuda()]
        token_embs = torch.cat([start_embs, end_embs], dim=1)
        return token_embs

    def get_seq_labels(self, example):
        label_list = []
        for arg_list in example["args"]:
            labels = [0] * len(example["start_idx"])
            for arg_info in arg_list:
                token_idx, label_idx = arg_info[:2]
                labels[token_idx] = label_idx
            label_list.append(torch.tensor(labels).cuda())
        return torch.cat(label_list)

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

        unary_score = torch.unsqueeze(self.other.arg_mlp(token_embs), dim=1)

        pred_embs = torch.unsqueeze(pred_embs, dim=0).repeat(token_embs.shape[0], 1, 1)
        token_embs = torch.unsqueeze(token_embs, dim=1).repeat(1, pred_embs.shape[1], 1)

        pairwise_score = self.other.arg_pred_mlp(torch.cat([pred_embs, token_embs], dim=2))

        score = unary_score + pairwise_score
        loss = self.loss_fn(score.reshape(-1, len(LABELS)), seq_labels)

        if self.training:
            return loss
        else:
            # L x P
            prediction = torch.argmax(score, dim=2).tolist()
            argument_list = [[] for _ in prediction[0]]

            for token_idx, token_pred_list in enumerate(prediction):
                for pred_idx, arg_pred in enumerate(token_pred_list):
                    if arg_pred != 0:
                        argument_list[pred_idx].append((token_idx, arg_pred))

            return argument_list
