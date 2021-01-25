import torch.nn as nn
from transformers import BertTokenizer, AutoModel, BertConfig
from transformers import BertLayer
from kbp_2015_utils.constants import SPEAKER_TAGS


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', pretrained_model=None, finetune=False, max_training_segments=None,
                 add_speaker_tags=False, use_local_attention=False, num_local_heads=12, use_srl=True, **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.max_training_segments = max_training_segments
        self.finetune = finetune
        self.use_srl = use_srl

        # Summary Writer
        if pretrained_model == 'bert':
            model_name = f'bert-{model_size}-cased'
        else:
            model_name = f'SpanBERT/spanbert-{model_size}-cased'

        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=False, gradient_checkpointing=(True if finetune else False))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if add_speaker_tags and self.finetune:
            print("Adding additional SPEAKER token")
            self.tokenizer.add_special_tokens({'additional_special_tokens': SPEAKER_TAGS})
            self.bert.resize_token_embeddings(len(self.tokenizer))

        self.use_local_attention = use_local_attention
        if self.use_local_attention or self.use_srl:
            bert_config = BertConfig.from_pretrained(model_name)
            if self.use_srl:
                bert_config.num_attention_heads = 12
            else:
                if num_local_heads > 0:
                    bert_config.num_attention_heads = num_local_heads
                else:
                    bert_config.num_attention_heads = 12
            self.additional_layer = BertLayer(bert_config)
            self.layer_norm = nn.LayerNorm(bert_config.hidden_size)

        self.pad_token = 0

        if not finetune:
            for param in self.bert.parameters():
                # Don't update BERT params
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        self.hsize = bert_hidden_size
