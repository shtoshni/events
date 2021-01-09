import torch
import random
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.base_encoder import BaseDocEncoder
import math


class IndependentDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(IndependentDocEncoder, self).__init__(**kwargs)

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
        if self.training and self.max_training_segments is not None:
            example = self.truncate_document(example)
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

    # @staticmethod
    def local_self_attention(self, example, encoded_doc):
        num_tokens = encoded_doc.shape[0]
        denom = math.sqrt(encoded_doc.shape[1])
        attention_mask = torch.zeros((num_tokens, num_tokens)).to(encoded_doc.device)
        sentence_map = example["sentence_map"]
        sentence_map_tens = torch.tensor(sentence_map).to(encoded_doc.device)
        min_sent_idx, max_sent_idx = sentence_map[0], sentence_map[-1]
        for sent_idx in range(min_sent_idx, max_sent_idx + 1):
            sent_idx_iden = torch.unsqueeze((sentence_map_tens == sent_idx).float(), dim=1)
            attention_mask += sent_idx_iden * torch.transpose(sent_idx_iden, 0, 1)

            # Add additional attention to neigboring sentence
            # sent_idx_next = torch.unsqueeze((sentence_map_tens == sent_idx + 1).float(), dim=1)
            # attention_mask += sent_idx_iden * torch.transpose(sent_idx_next, 0, 1)

        assert (torch.max(attention_mask) == 1.0)
        # pairwise_sim = torch.matmul(self.proj_query(encoded_doc), self.proj_key(encoded_doc).t())/denom
        # pairwise_sim = pairwise_sim * attention_mask + (1 - attention_mask) * (-1e10)
        # encoded_doc = torch.matmul(torch.softmax(pairwise_sim, dim=1), self.proj_val(encoded_doc))

        pairwise_sim = torch.matmul(encoded_doc, encoded_doc.t()) / denom
        pairwise_sim = pairwise_sim * attention_mask + (1 - attention_mask) * (-1e10)
        encoded_doc = torch.matmul(torch.softmax(pairwise_sim, dim=1), encoded_doc)

        return encoded_doc

    def truncate_document(self, example):
        sentences = example["sentences"]
        num_sentences = len(example["sentences"])

        if num_sentences > self.max_training_segments and self.max_training_segments is not None:
            sentence_offset = random.randint(0, num_sentences - self.max_training_segments)
            word_offset = sum([len(sent) for sent in sentences[:sentence_offset]])
            sentences = sentences[sentence_offset: sentence_offset + self.max_training_segments]
            num_words = sum([len(sent) for sent in sentences])
            sentence_map = example["sentence_map"][word_offset: word_offset + num_words]
            subtoken_map = example["subtoken_map"][word_offset: word_offset + num_words]

            clusters = []
            for orig_cluster in example["clusters"]:
                cluster = []
                for ment_start, ment_end, ment_type in orig_cluster:
                    if ment_end >= word_offset and ment_start < word_offset + num_words:
                        cluster.append((ment_start - word_offset, ment_end - word_offset, ment_type))

                if len(cluster):
                    clusters.append(cluster)

            example["sentences"] = sentences
            example["clusters"] = clusters
            example["sentence_map"] = sentence_map
            example["subtoken_map"] = subtoken_map

            return example
        else:
            return example

    def forward(self, example):
        example, doc_tens, sent_len_list = self.tensorize_example(example)
        encoded_doc = self.encode_doc(doc_tens, sent_len_list)
        if self.use_local_attention:
            encoded_doc = self.local_self_attention(example, encoded_doc)
        return encoded_doc
