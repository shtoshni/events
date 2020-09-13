import torch
import torch.nn as nn
from os import path
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from transformers import BertModel, BertTokenizer
from red_utils.constants import DUPLICATE_START_TAG, DUPLICATE_END_TAG


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', pretrained_bert_dir=None, finetune=False, **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.last_layers = 1
        self.finetune = finetune

        # Summary Writer
        if pretrained_bert_dir:
            self.bert = BertModel.from_pretrained(
                path.join(pretrained_bert_dir, "spanbert_{}".format(model_size)), output_hidden_states=True,
                # gradient_checkpointing=False
                gradient_checkpointing=(True if finetune else False)
            )
        else:
            bert_model_name = 'bert-' + model_size + '-cased'
            self.bert = BertModel.from_pretrained(
                bert_model_name, output_hidden_states=True, gradient_checkpointing=(True if finetune else False))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.add_tokens([DUPLICATE_START_TAG, DUPLICATE_END_TAG])
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.pad_token = 0

        if not finetune:
            for param in self.bert.parameters():
                # Don't update BERT params
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        self.hsize = self.last_layers * bert_hidden_size
