import os
import re
import sys
import json
import collections
import truecase
from collections import defaultdict, OrderedDict
import xml
import xml.etree.ElementTree as ET

from os import path
from transformers import BertTokenizer
from data_processing.utils import get_all_ent_info
from red_utils.constants import ELEM_TYPE_TO_IDX
from red_utils.constants import DUPLICATE_START_TAG, DUPLICATE_END_TAG


BERT_RE = re.compile(r'## *')


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.segment_info = []
        self.mentions = []

    def get_span_to_type(self, ent_id_to_info):
        span_to_type = {}
        for ent_id in ent_id_to_info:
            span_start, span_end, ent_type = ent_id_to_info[ent_id]
            span_ends = tuple([span_start, span_end])

            if span_ends in span_to_type:
                span_to_type[span_ends] = ELEM_TYPE_TO_IDX['BOTH']#, ELEM_TYPE_TO_IDX[ent_type])
            else:
                span_to_type[span_ends] = ELEM_TYPE_TO_IDX[ent_type]

        return span_to_type

    def finalize(self, ent_id_to_info):
        span_to_type = self.get_span_to_type(ent_id_to_info)
        # span_ends_seen = set()
        for ent_id in ent_id_to_info:
            span_start, span_end, ent_type = ent_id_to_info[ent_id]
            span_ends = tuple([span_start, span_end])
            # if span_ends in span_ends_seen:
            #     continue
            # else:
            #     span_ends_seen.add(span_ends)
            # span_type = span_to_type[span_ends]
            mention = (span_start, span_end, ELEM_TYPE_TO_IDX[ent_type])
            self.mentions.append(mention)

        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        assert len(self.mentions) == len(set(self.mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "mentions": self.mentions,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
        }


def flatten(l):
    return [item for sublist in l for item in sublist]


def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 2,
                  len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(current + max_segment_len - 1 - 2,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")
        document_state.segments.append(document_state.subtokens[current:end + 1])
        subtoken_map = document_state.subtoken_map[current: end + 1]
        document_state.segment_subtoken_map.append(subtoken_map)
        current = end + 1


def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) for s in segments])
    for segment in segments:
        for i in range(len(segment)):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
    return sent_map


def get_document(doc_name, tokenized_doc, ent_id_to_info, segment_len):
    document_state = DocumentState(doc_name)
    word_idx = -1
    for idx, token in enumerate(tokenized_doc):
        if not BERT_RE.match(token):
            word_idx += 1

        document_state.tokens.append(token)
        # Subtoken and token are same
        document_state.subtokens.append(token)
        if idx == len(tokenized_doc) - 1:
            # End of document
            document_state.token_end += ([True])
        else:
            next_token = tokenized_doc[idx + 1]
            if BERT_RE.match(next_token):
                # If the next token has ## at the start then the current subtoken
                # is clearly not the end of the token
                document_state.token_end += ([False])
            else:
                document_state.token_end += ([True])

        # No annotation of sentence boundaries in RED. Hence always False for sentence end.
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)

    split_into_segments(document_state, segment_len,
                        document_state.sentence_end, document_state.token_end)
    document = document_state.finalize(ent_id_to_info)
    return document


def tokenize_span(span, doc_name, tokenizer):
    if "proxy/" in doc_name:
        span = truecase.get_true_case(span)
    return tokenizer.tokenize(span)


def tokenize_doc(doc_name, tokenizer, source_file, annotation_file):
    # Read the source document
    # orig_doc_str = "".join(open(source_file).readlines())
    doc_str = "".join(open(source_file).readlines())

    # Parse the XML
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # Get entity information from the annotation file
    ent_map, ent_list = get_all_ent_info(root)

    tokenized_doc = []
    token_counter = 0
    char_offset = 0  # Till what point has the document been processed
    ent_id_to_info = OrderedDict()

    real_span_to_tokenized_span = {}
    for (span_start, span_end), ent_type, ent_id in ent_list:
        # Tokenize the string before the span and after the last span
        real_span = tuple([span_start, span_end])
        if real_span not in real_span_to_tokenized_span:
            before_span_str = doc_str[char_offset: span_start]
            before_span_tokens = tokenize_span(before_span_str, doc_name, tokenizer)
            tokenized_doc.extend(before_span_tokens)
            token_counter += len(before_span_tokens)

            # Tokenize the span
            span_tokens = tokenize_span(doc_str[span_start: span_end], doc_name, tokenizer)

            if ent_type != 'DUPLICATE':
                ent_id_to_info[ent_id] = tuple([
                    token_counter, token_counter + len(span_tokens) - 1, ent_type])
                real_span_to_tokenized_span[real_span] = tuple(
                    [token_counter, token_counter + len(span_tokens) - 1])
            else:
                # No need to include the DUPLICATE span in ent_id_to_info
                real_span_to_tokenized_span[real_span] = tuple(
                    [token_counter, token_counter + len(span_tokens) - 1 + 2])
                # Add the tokenized doc
                span_tokens = [DUPLICATE_START_TAG] + span_tokens + [DUPLICATE_END_TAG]

            tokenized_doc.extend(span_tokens)
            char_offset = span_end
            token_counter += len(span_tokens)
        else:
            tokenized_start, tokenized_end = real_span_to_tokenized_span[real_span]
            ent_id_to_info[ent_id] = tuple([tokenized_start, tokenized_end, ent_type])

    # Add the tokens after the last span
    rem_doc = doc_str[char_offset:]
    rem_tokens = tokenize_span(rem_doc, doc_name, tokenizer)
    token_counter += len(rem_tokens)

    tokenized_doc.extend(rem_tokens)
    return tokenized_doc, ent_id_to_info


def minimize_partition(split, tokenizer,
                       source_dir, ann_dir, doc_dir, seg_len, output_dir):
    split_list_file = path.join(doc_dir, f'{split}.txt')
    split_files = set([file_name.strip() for file_name in open(split_list_file).readlines()])

    source_files, annotation_files = [], []
    for split_file in split_files:
        source_files.append(path.join(source_dir, split_file))
        annotation_files.append(path.join(ann_dir, split_file) + ".RED-Relation.gold.completed.xml")

    output_path = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))
    count = 0

    print("Minimizing {}".format(split))
    with open(output_path, "w") as output_file:
        for doc_name, source_file, annotation_file in \
                zip(split_files, source_files, annotation_files):
            tokenized_doc, ent_id_to_info = tokenize_doc(doc_name, tokenizer, source_file, annotation_file)
            document = get_document(doc_name, tokenized_doc, ent_id_to_info, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(source_dir, ann_dir, doc_dir, seg_len, output_dir):
    # do_lower_case = True if 'chinese' in vocab_file else False
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # Create cross validation output dir

    minimize_partition("dev", tokenizer, source_dir, ann_dir, doc_dir, seg_len, output_dir)
    minimize_partition("train", tokenizer, source_dir, ann_dir, doc_dir, seg_len, output_dir)
    minimize_partition("test", tokenizer, source_dir, ann_dir, doc_dir, seg_len, output_dir)


if __name__ == "__main__":
    input_dir = sys.argv[1]

    source_dir = path.join(input_dir, "data/source")
    ann_dir = path.join(input_dir, "data/mod_annotation")
    doc_dir = path.join(input_dir, "docs")

    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for seg_len in [512]:#[128, 256, 384, 512]:
        minimize_split(source_dir, ann_dir, doc_dir, seg_len, output_dir)
