import torch.nn as nn
from os import path
from transformers import BertModel, BertTokenizer, AutoModel
from kbp_2015_utils.constants import SPEAKER_TAGS


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', pretrained_bert_dir=None, finetune=False, max_training_segments=None,
                 add_speaker_tags=False, use_local_attention=False, **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.max_training_segments = max_training_segments
        self.finetune = finetune

        # Summary Writer
        if pretrained_bert_dir:
            self.bert = AutoModel.from_pretrained(
                path.join(pretrained_bert_dir, "spanbert_{}".format(model_size)), output_hidden_states=False,
                # gradient_checkpointing=False
                gradient_checkpointing=(True if finetune else False)
            )
        else:
            bert_model_name = 'bert-' + model_size + '-cased'
            self.bert = AutoModel.from_pretrained(
                bert_model_name, output_hidden_states=False, gradient_checkpointing=(True if finetune else False))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        if add_speaker_tags and self.finetune:
            print("Adding additional SPEAKER token")
            self.tokenizer.add_special_tokens({'additional_special_tokens': SPEAKER_TAGS})
            self.bert.resize_token_embeddings(len(self.tokenizer))

        self.use_local_attention = use_local_attention

        self.pad_token = 0

        if not finetune:
            for param in self.bert.parameters():
                # Don't update BERT params
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        self.hsize = bert_hidden_size
        if self.use_local_attention:
            self.hsize = 2 * self.hsize
