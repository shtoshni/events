import os
import re
import sys
import json
import glob
from collections import OrderedDict
import xml.etree.ElementTree as ET
import truecase
from cleantext import clean

from os import path
from transformers import BertTokenizer
from data_processing.kbp_2015.utils import parse_ann_file
import argparse
from kbp_2015_utils.constants import SPLIT_TO_DIR, SUBDIR_DICT, SUBDIR_EXT

BERT_RE = re.compile(r'## *')
NEWLINE_TOKEN = "[NEWL]"


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
        self.clusters = []

    def finalize(self, clusters, ent_id_to_info):
        # populate clusters
        processed_ent_ids = set()
        for cluster in clusters:
            cur_cluster = []
            for ent_id in cluster:
                processed_ent_ids.add(ent_id)
                cur_cluster.append(ent_id_to_info[ent_id])
            self.clusters.append(cur_cluster)

        all_mentions = flatten(self.clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        # assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.clusters,
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


def get_document(doc_name, tokenized_doc, clusters, ent_id_to_info, segment_len):
    document_state = DocumentState(doc_name)
    word_idx = -1
    for idx, token in enumerate(tokenized_doc):
        if token == NEWLINE_TOKEN:
            # [NEWL] corresponds to "\n" in real doc
            document_state.sentence_end[-1] = True
            continue

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

        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    # Last word in the document is obviously end of sentence
    document_state.sentence_end[-1] = True
    split_into_segments(document_state, segment_len,
                        document_state.sentence_end, document_state.token_end)
    document = document_state.finalize(clusters, ent_id_to_info)
    return document


def tokenize_span(span, tokenizer, doc_name):
    span = span.strip()
    if span == '':
        return [], 0

    span = span.replace("\n", "<N>")
    span = span.replace("<N>", NEWLINE_TOKEN)
    newline_count = span.count(NEWLINE_TOKEN)

    tokenized_span = tokenizer.tokenize(span)

    if "[UNK]" in tokenized_span:
        # Try cleaning the text and reprocess
        cleaned_span = clean(span, lower=False)
        tokenized_span = tokenizer.tokenize(cleaned_span)
        if "[UNK]" in tokenized_span:
            print(span, tokenized_span, doc_name)
            sys.exit()

    return tokenized_span, newline_count


def tokenize_doc(doc_name, source_file, ann_file, tokenizer):
    # Read the source document
    doc_str = "".join(open(source_file, encoding="utf-8").readlines())
    doc_str = doc_str.replace('’', "'")
    # altered_doc_str = doc_str.replace("\n", "[NEWL]")

    # Parse the XML
    mention_list, clusters = parse_ann_file(ann_file)
    # Sort mentions by their starting and ending point - Priority to starting point
    mention_list = sorted(mention_list, key=lambda x: x[0] + 1e-5 * x[1])

    tokenized_doc = []
    token_counter = 0
    char_offset = 0  # Till what point has the document been processed
    ent_id_to_info = OrderedDict()

    real_span_to_tokenized_span = {}
    char_to_tokenized_idx = {}
    for span_start, span_end, mention_info in mention_list:
        # Tokenize the string before the span and after the last span
        real_span = tuple([span_start, span_end])
        if real_span not in real_span_to_tokenized_span:
            if char_offset > span_start:
                # Overlapping span
                relv_tokens, _ = tokenize_span(doc_str[span_start: span_end], tokenizer, doc_name)

                if span_start in char_to_tokenized_idx:
                    # The shorted span has already been processed! "x" "x y" overlap
                    counter_idx, token_idx = char_to_tokenized_idx[span_start]
                    remaining_tokens, newline_count = tokenize_span(doc_str[char_offset: span_end], tokenizer, doc_name)
                    tokenized_doc.extend(remaining_tokens)
                    char_offset = span_end
                    # Don't count newline tokens as they will ultimately be removed.
                    token_counter += len(remaining_tokens) - newline_count
                    char_to_tokenized_idx[char_offset] = (token_counter, len(tokenized_doc))

                    ent_id_to_info[mention_info["id"]] = tuple([counter_idx, counter_idx + len(relv_tokens) - 1,
                                                                mention_info])

                elif span_end in char_to_tokenized_idx:
                    # The span has already been processed! "x y" "y" overlap
                    token_idx, counter_idx = char_to_tokenized_idx[span_end]
                    proc_tokens = tokenized_doc[counter_idx - len(relv_tokens): counter_idx]

                    try:
                        assert(relv_tokens == proc_tokens)
                    except AssertionError:
                        print(doc_name, relv_tokens, proc_tokens)

                    ent_id_to_info[mention_info["id"]] = tuple([counter_idx - len(relv_tokens), counter_idx - 1,
                                                                mention_info])
                else:
                    # We are at lord's mercy
                    print("JESUS")
                    raise ValueError(doc_name)
            else:
                before_span_str = doc_str[char_offset: span_start]
                before_span_tokens, newline_count = tokenize_span(before_span_str, tokenizer, doc_name)
                tokenized_doc.extend(before_span_tokens)

                # Don't count newline tokens as they will ultimately be removed.
                token_counter += len(before_span_tokens) - newline_count
                char_to_tokenized_idx[span_start] = (token_counter, len(tokenized_doc))

                # Tokenize the span
                span_tokens, newline_count = tokenize_span(doc_str[span_start: span_end], tokenizer, doc_name)

                ent_id_to_info[mention_info["id"]] = tuple([
                        token_counter, token_counter + len(span_tokens) - 1, mention_info])

                real_span_to_tokenized_span[real_span] = tuple(
                    [token_counter, token_counter + len(span_tokens) - 1])

                tokenized_doc.extend(span_tokens)
                char_offset = span_end
                # Don't count newline tokens as they will ultimately be removed.
                token_counter += len(span_tokens) - newline_count
                char_to_tokenized_idx[char_offset] = (token_counter, len(tokenized_doc))
        else:
            tokenized_start, tokenized_end = real_span_to_tokenized_span[real_span]
            ent_id_to_info[mention_info["id"]] = tuple([tokenized_start, tokenized_end, mention_info])

    # Add the tokens after the last span
    rem_doc = doc_str[char_offset:]
    rem_tokens, newline_count = tokenize_span(rem_doc, tokenizer, doc_name)
    # Don't count newline tokens as they will ultimately be removed.
    token_counter += len(rem_tokens) - newline_count

    tokenized_doc.extend(rem_tokens)
    return tokenized_doc, ent_id_to_info, clusters


def minimize_partition(args, split, tokenizer, seg_len):
    split_dir = path.join(args.input_dir, SPLIT_TO_DIR[split])
    source_dir = path.join(split_dir, SUBDIR_DICT["source"])
    ann_dir = path.join(split_dir, SUBDIR_DICT["ann"])

    source_files = sorted(glob.glob(path.join(source_dir, "*" + SUBDIR_EXT["source"])))
    # print(source_files, path.join(source_dir, SUBDIR_EXT["source"]))
    doc_ids, ann_files = [], []
    for source_file in source_files:
        doc_id = path.splitext(path.basename(source_file))[0]
        doc_ids.append(doc_id)
        ann_files.append(path.join(ann_dir, doc_id + SUBDIR_EXT["ann"]))

    output_path = path.join(args.output_dir, "{}.{}.jsonlines".format(split, seg_len))
    count = 0
    print("Minimizing {}".format(split))
    with open(output_path, "w") as output_file:
        for doc_name, source_file, annotation_file in \
                zip(doc_ids, source_files, ann_files):
            tokenized_doc, ent_id_to_info, clusters = tokenize_doc(
                doc_name, source_file, annotation_file, tokenizer)
            document = get_document(doc_name, tokenized_doc, clusters, ent_id_to_info, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args, seg_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': [NEWLINE_TOKEN]})
    for split in ["dev", "test", "train"]:
        minimize_partition(args, split, tokenizer, seg_len)


if __name__ == "__main__":
    input_dir = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory root")
    parser.add_argument("output_dir", type=str, help="Output directory")

    parsed_args = parser.parse_args()

    if not os.path.isdir(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)
    for window_len in [512]:
        minimize_split(parsed_args, window_len)
