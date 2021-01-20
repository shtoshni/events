import torch
import torch.nn as nn
from pytorch_utils.utils import get_sequence_mask
from transformers import BertTokenizer, AutoModel, AdapterType
from os import path


class IndependentDocEncoder(nn.Module):
    def __init__(self, model_size='base', pretrained_bert_dir=None, finetune=False, **kwargs):
        super(IndependentDocEncoder, self).__init__()
        self.finetune = finetune

        # Summary Writer
        if pretrained_bert_dir:
            self.bert = AutoModel.from_pretrained(
                path.join(pretrained_bert_dir, "spanbert_{}".format(model_size)), output_hidden_states=False,
            )

        else:
            bert_model_name = 'bert-' + model_size + '-cased'
            self.bert = AutoModel.from_pretrained(
                bert_model_name, output_hidden_states=False, gradient_checkpointing=(True if finetune else False))

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.pad_token = 0

        if not finetune:
            self.bert.add_adapter("srl", AdapterType.text_task)

            # for param in self.bert.parameters():
            #     # Don't update BERT params
            #     param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        self.hsize = bert_hidden_size

    def encode_doc(self, document, text_length_list):
        """
        Encode chunks of a document.
        batch_excerpt: C x L where C is number of chunks padded upto max length of L
        text_length_list: list of length of chunks (length C)
        """
        num_chunks = len(text_length_list)
        attn_mask = get_sequence_mask(torch.tensor(text_length_list).cuda()).cuda().float()
        # attn_mask = attn_mask.clone().detach().requires_grad_(True)

        if not self.finetune:
            with torch.no_grad():
                outputs = self.bert(document, attention_mask=attn_mask)  # C x L x E
        else:
            outputs = self.bert(document, attention_mask=attn_mask)  # C x L x E

        encoded_repr = outputs[0]

        unpadded_encoded_output = []
        for i in range(num_chunks):
            unpadded_encoded_output.append(
                # Remove CLS and SEP from token embeddings
                encoded_repr[i, 1:text_length_list[i] - 1, :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)
        encoded_output = encoded_output
        return encoded_output

    def tensorize_example(self, example):
        sentences = example["sentences"]
        sent_len_list = [(len(sent) + 2) for sent in sentences]

        max_sent_len = max(sent_len_list)
        # print(max_sent_len)
        # Add 0 and 1 for CLS and SEP
        padded_sent = [[101] + self.tokenizer.convert_tokens_to_ids(sent) + [102]
                       + [self.pad_token] * (max_sent_len - (len(sent) + 2))
                       for sent in sentences]
        doc_tens = torch.tensor(padded_sent).cuda()
        return example, doc_tens, sent_len_list

    def forward(self, example):
        example, doc_tens, sent_len_list = self.tensorize_example(example)
        encoded_doc = self.encode_doc(doc_tens, sent_len_list)
        return encoded_doc
