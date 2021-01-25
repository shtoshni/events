import torch
import random
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.base_encoder import BaseDocEncoder
import math
from srl.constants import LABELS


class IndependentDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(IndependentDocEncoder, self).__init__(**kwargs)

    def encode_doc(self, example, document, text_length_list):
        num_chunks = len(text_length_list)
        attn_mask = get_sequence_mask(torch.tensor(text_length_list).cuda()).cuda().float()

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
        output = (encoded_output,)

        if self.use_srl or self.use_local_attention:
            loss, weight = 0.0, 0.0
            arg_output = []

            for idx, (encoded_window, attention_mask) in enumerate(zip(
                    unpadded_encoded_output, example["local_attention_mask"])):
                # Detach the output of BERT
                window_output = self.additional_layer(
                    torch.unsqueeze(encoded_window, dim=0),
                    output_attentions=True, attention_mask=attention_mask[None, None, :, :])

                arg_output.append(torch.squeeze(window_output[0], dim=0))

                if self.use_srl:
                    gt_attn_map = example["srl_attention_map"][idx]
                    attention = window_output[1]
                    attention = torch.squeeze(attention, dim=0)[:6]
                    # print(torch.sum())

                    # term = torch.log(torch.sum(attention * gt_attn_map, dim=-1) + 1e-8) * torch.sum(gt_attn_map, dim=-1)
                    # loss -= torch.sum(term)
                    attention = attention/(torch.sum(attention, dim=-1, keepdim=True) + 1e-10)
                    # print(attention.shape)
                    loss -= torch.sum((torch.log(attention + 1e-10) - torch.log(gt_attn_map + 1e-10)) * gt_attn_map)

                    # loss += torch.norm((attention - gt_attn_map) * (gt_attn_map > 0).float(), p='fro') ** 2
                    weight += torch.sum(gt_attn_map)

            # print(loss/weight)
            arg_output = torch.cat(arg_output, dim=0)  # self.layer_norm(torch.cat(arg_output, dim=0) + encoded_output)
            output = (encoded_output, arg_output,)

            # return output + (arg_output, loss/weight,)
            # arg_output = torch.cat(arg_output, dim=0)
            # output = (torch.cat([encoded_output, arg_output], dim=-1),)
            if self.use_srl:
                output = output + (loss / weight,)

        return output

    def tensorize_example(self, example):
        if self.training and self.max_training_segments is not None:
            example = self.truncate_document(example)

        if self.use_srl:
            example["srl_attention_map"] = self.construct_srl_attention_map(example)

        if self.use_srl or self.use_local_attention:
            example["local_attention_mask"] = self.local_attention_mask(example)

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

    @staticmethod
    def local_attention_mask(example):
        attention_mask_list = []
        doc_offset = 0
        for window_idx, sentence in enumerate(example["sentences"]):
            attention_mask = torch.zeros(len(sentence), len(sentence)).cuda()
            sentence_map = example["sentence_map"][doc_offset: doc_offset + len(sentence)]
            sentence_map_tens = torch.tensor(sentence_map).cuda()
            min_sent_idx, max_sent_idx = sentence_map[0], sentence_map[-1]
            for sent_idx in range(min_sent_idx, max_sent_idx + 1):
                sent_idx_iden = torch.unsqueeze((sentence_map_tens == sent_idx).float(), dim=1)
                attention_mask += sent_idx_iden * torch.transpose(sent_idx_iden, 0, 1)

            for offset in range(-10, 11):
                attention_mask += torch.diag(
                    torch.ones(len(sentence)).cuda(), diagonal=offset)[:len(sentence), :len(sentence)]

            attention_mask = torch.clamp(attention_mask, min=0, max=1)

            attention_mask_list.append((1 - attention_mask) * -1e4)
            doc_offset += len(sentence)

        return attention_mask_list

    @staticmethod
    def construct_srl_attention_map(example):
        attention_maps_list = []
        doc_offset = 0
        for sent_idx, sentence in enumerate(example["sentences"]):
            sent_attention_maps = []
            for label_idx in range(1, len(LABELS)):
                attn_map = torch.zeros(len(sentence), len(sentence)).cuda()
                key = str(label_idx)
                if key in example["srl_info"]:
                    pred_arg_list = example["srl_info"][key]
                    for pred_arg_info in pred_arg_list:
                        if pred_arg_info[4] == sent_idx:
                            pred_start, pred_end, arg_start, arg_end = pred_arg_info[:4]
                            arg_len = arg_end - arg_start + 1
                            for pred_idx in range(pred_start, pred_end + 1):
                                for arg_idx in range(arg_start, arg_end + 1):
                                    try:
                                        attn_map[pred_idx - doc_offset, arg_idx - doc_offset] = 1.0/arg_len
                                    except IndexError:
                                        import sys
                                        sys.exit()
                sent_attention_maps.append(attn_map)

            # Sentence processed
            doc_offset += len(sentence)
            sent_attn_map = torch.stack(sent_attention_maps, dim=0)
            # print(sent_attn_map.shape)
            attention_maps_list.append(sent_attn_map)

        return attention_maps_list

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

            if self.use_srl:
                srl_info_dict = {}
                for key, pred_arg_list in example["srl_info"].items():
                    mod_pred_arg_list = []
                    for boundary in pred_arg_list:
                        pred_start, pred_end, arg_start, arg_end = boundary[:4]
                        if arg_end >= word_offset and arg_start < word_offset + num_words:
                            if pred_end >= word_offset and pred_start < word_offset + num_words:
                                mod_pred_arg_list.append([
                                    pred_start - word_offset, pred_end - word_offset,
                                    arg_start - word_offset, arg_end - word_offset, boundary[4] - sentence_offset])

                    if len(mod_pred_arg_list):
                        srl_info_dict[key] = mod_pred_arg_list

                example["srl_info"] = srl_info_dict

            example["sentences"] = sentences
            example["clusters"] = clusters
            example["sentence_map"] = sentence_map
            example["subtoken_map"] = subtoken_map

            return example
        else:
            return example

    def forward(self, example):
        example, doc_tens, sent_len_list = self.tensorize_example(example)
        output = self.encode_doc(example, doc_tens, sent_len_list)

        return output
