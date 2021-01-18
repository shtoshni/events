import os
import re
import sys
import json
import glob
from collections import OrderedDict

from os import path
from transformers import BertTokenizer
from data_processing.kbp_2015.utils import parse_ann_file
import argparse
from kbp_2015_utils.constants import SPLIT_TO_DIR, SUBDIR_DICT, SUBDIR_EXT, SPEAKER_TAGS

BERT_RE = re.compile(r'## *')
NEWLINE_TOKEN = "[NEWL]"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.add_special_tokens(
    # SPEAKER_TAGS are added to final tokenized doc but NEWLINE_TOKEN is only for tracking of new lines.
    {'additional_special_tokens': [NEWLINE_TOKEN] + SPEAKER_TAGS})


class DocumentState(object):
    def __init__(self, key, doc_type):
        self.doc_key = key
        self.doc_type = doc_type
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
        self.orig_doc = None
        self.proc_doc = None
        self.tokenized_sentences = None
        self.token_idx_to_orig_span_start = None
        self.token_idx_to_orig_span_end = None

    def finalize(self, clusters, ent_id_to_info):
        # populate clusters
        processed_ent_ids = set()
        for cluster in clusters:
            cur_cluster = []
            for ent_id in cluster:
                processed_ent_ids.add(ent_id)
                try:
                    cur_cluster.append(ent_id_to_info[ent_id])
                except KeyError:
                    continue
            if len(cur_cluster):
                self.clusters.append(cur_cluster)

        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        # assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "doc_type": self.doc_type,
            "sentences": self.segments,
            "tokenized_sentences": self.tokenized_sentences,
            "clusters": self.clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            "orig_doc": self.orig_doc,
            "proc_doc": self.proc_doc,
            "token_idx_to_orig_span_start": self.token_idx_to_orig_span_start,
            "token_idx_to_orig_span_end": self.token_idx_to_orig_span_end,
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


def get_document(doc_name, segment_len, info_dict):
    document_state = DocumentState(doc_name, info_dict["doc_type"])
    document_state.orig_doc = info_dict["orig_doc"]
    document_state.proc_doc = info_dict["proc_doc"]
    document_state.tokenized_sentences = info_dict["tokenized_sentences"]
    document_state.token_idx_to_orig_span_start = info_dict["token_idx_to_orig_span_start"]
    document_state.token_idx_to_orig_span_end = info_dict["token_idx_to_orig_span_end"]

    word_idx = -1
    tokenized_doc = info_dict["tokenized_doc"]
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
    document = document_state.finalize(info_dict["clusters"], info_dict["ent_id_to_info"])

    return document


def read_source_doc(proc_source_file, no_speaker_tags=False):
    proc_doc = json.loads(open(proc_source_file).read())

    orig_span_start_to_token_idx = dict()
    orig_span_end_to_token_idx = dict()

    token_idx_to_orig_span_start = dict()
    token_idx_to_orig_span_end = dict()

    proc_doc_str = " "
    cur_speaker = None
    tokenized_doc = []
    newline_counter = 0
    tokenized_sentences = []

    for sentence in proc_doc["sentences"]:
        tokens = sentence["tokens"]
        # Verified that each processed sentence has a unique speaker. So the segmentation is fine.
        # We can just append the speaker tag before the start of sentence
        if "speaker" in tokens[0] and not no_speaker_tags:
            if cur_speaker != tokens[0]['speaker']:
                cur_speaker = tokens[0]['speaker']
                speaker_str = f"{SPEAKER_TAGS[0]} {tokens[0]['speaker']} {SPEAKER_TAGS[1]} "
                proc_doc_str += speaker_str

                speaker_tokens = tokenizer.basic_tokenizer.tokenize(
                    speaker_str, never_split=tokenizer.all_special_tokens)
                # Speaker tokens are a modification of original text and don't correspond to
                # any actual token. Since they don't actually correspond to any mention we can just
                # set it to some value. Here we just set it current document offset
                for basic_token in speaker_tokens:
                    # token_idx_to_orig_span_start[len(tokenized_doc) - newline_counter] = None

                    if basic_token in tokenizer.all_special_tokens:
                        tokenized_doc.append(basic_token)
                    else:
                        for subword_token in tokenizer.wordpiece_tokenizer.tokenize(basic_token):
                            tokenized_doc.append(subword_token)
                    # token_idx_to_orig_span_end[len(tokenized_doc) - 1 - newline_counter] = None

        new_tokenized_sentence = []
        for token in tokens:
            token_text = token["word"]
            if (len(token_text) > 1) and (token_text == token_text.upper()):
                # Word is all caps. Use truecase output instead
                token_text = token["truecaseText"]
            basic_tokens = tokenizer.basic_tokenizer.tokenize(token_text)

            # The original span might have spaces in it like "2 1/2" or "(850) 224-7263"
            # In such cases, the sum won't be equal to parts.
            # The idea behind this boolean is that in case this sum is equal to part, we can do a finer mapping
            # of orig_doc_offset_to_token_idx calculation.
            is_sum_equal_to_part = (len(token_text) == sum([len(basic_token) for basic_token in basic_tokens]))
            orig_doc_offset = token["characterOffsetBegin"]

            if is_sum_equal_to_part:
                for basic_token in basic_tokens:
                    # len(tokenized_doc) because at least a token will be added and that token idx
                    # corresponds to span start
                    start_token_idx = len(tokenized_doc) - newline_counter
                    orig_span_start_to_token_idx[orig_doc_offset] = start_token_idx

                    for subword_token in tokenizer.wordpiece_tokenizer.tokenize(basic_token):
                        tokenized_doc.append(subword_token)
                    # len(tokenized_doc) - 1 because the index is less than 1
                    end_token_idx = len(tokenized_doc) - 1 - newline_counter
                    orig_span_end_to_token_idx[orig_doc_offset + len(basic_token)] = end_token_idx

                    # More fine-grained original document offsets possible
                    new_tokenized_sentence.append([basic_token, (start_token_idx, end_token_idx)])
                    orig_doc_offset += len(basic_token)

            else:
                orig_span_start_to_token_idx[orig_doc_offset] = len(tokenized_doc) - newline_counter
                for basic_token in basic_tokens:
                    start_token_idx = len(tokenized_doc) - newline_counter
                    for subword_token in tokenizer.wordpiece_tokenizer.tokenize(basic_token):
                        tokenized_doc.append(subword_token)

                    end_token_idx = len(tokenized_doc) - 1 - newline_counter
                    new_tokenized_sentence.append([basic_token, (start_token_idx, end_token_idx)])

                orig_span_end_to_token_idx[token["characterOffsetEnd"]] = len(tokenized_doc) - 1 - newline_counter

            # orig_doc_offset = token["characterOffsetEnd"]
            proc_doc_str += token_text + " "

        tokenized_sentences.append(new_tokenized_sentence)
        proc_doc_str += "\n"
        tokenized_doc.append(NEWLINE_TOKEN)
        newline_counter += 1

    # Add the reverse entries to token_idx to document offsets
    for key, value in orig_span_start_to_token_idx.items():
        token_idx_to_orig_span_start[value] = key

    for key, value in orig_span_end_to_token_idx.items():
        token_idx_to_orig_span_end[value] = key

    return (proc_doc_str, tokenized_doc, tokenized_sentences, orig_span_start_to_token_idx,
            orig_span_end_to_token_idx, token_idx_to_orig_span_start, token_idx_to_orig_span_end)


def tokenize_doc(doc_name, source_file, proc_source_file, ann_file, no_speaker_tags=False):
    # Read the source document
    orig_doc_str = open(source_file).read()
    orig_doc_str = orig_doc_str.replace("\n", " ")
    # orig_doc_str = orig_doc_str.replace('’', "'")

    proc_doc_str, tokenized_doc, tokenized_sentences, orig_span_start_to_token_idx, orig_span_end_to_token_idx, \
        token_idx_to_orig_span_start, token_idx_to_orig_span_end = read_source_doc(proc_source_file, no_speaker_tags=no_speaker_tags)

    tokenized_doc_without_newl = [token for token in tokenized_doc if token != NEWLINE_TOKEN]

    # Parse the XML
    doc_type, mention_list, clusters = parse_ann_file(ann_file)
    # Sort mentions by their starting and ending point - Priority to starting point
    mention_list = sorted(mention_list, key=lambda x: x[0] + 1e-5 * x[1])

    ent_id_to_info = OrderedDict()

    for span_start, span_end, mention_info in mention_list:
        if span_start in orig_span_start_to_token_idx and span_end in orig_span_end_to_token_idx:
            start_token_idx = orig_span_start_to_token_idx[span_start]
            end_token_idx = orig_span_end_to_token_idx[span_end]
            ent_id_to_info[mention_info["id"]] = tuple([start_token_idx, end_token_idx, mention_info])

            proc_span = tokenizer.convert_tokens_to_string(tokenized_doc_without_newl[start_token_idx: end_token_idx + 1])
            orig_span = orig_doc_str[span_start: span_end]
            try:
                assert (proc_span.replace(" ", "").lower() == orig_span.replace(" ", "").lower())
            except AssertionError:
                print(f"WARNING: {doc_name} - Proc span: ({proc_span}) is different from  original span ({orig_span})")
                pass
        else:
            if (span_end + 1) in orig_span_end_to_token_idx:
                # There are some mistakes in training data where instead of marking "responses" they marked "response"
                ent_id_to_info[mention_info["id"]] = tuple([
                    orig_span_start_to_token_idx[span_start], orig_span_end_to_token_idx[span_end + 1], mention_info])
            else:
                # Some of the spans in URLs were marked which were removed during preprocessing
                # There are other which are marked in the middle of a pharse - "antiwar" where "war" is marked
                print(f"{doc_name}: Found no matching token that matches the offset boundary for span "
                      f"{orig_doc_str[span_start: span_end]} that starts at {span_start} with context"
                      f"{orig_doc_str[span_start - 10: span_end + 10]}")

    return {"orig_doc": orig_doc_str, "proc_doc": proc_doc_str, "doc_type": doc_type,
            "tokenized_sentences": tokenized_sentences,
            "token_idx_to_orig_span_start": token_idx_to_orig_span_start,
            "token_idx_to_orig_span_end": token_idx_to_orig_span_end, "clusters": clusters,
            "tokenized_doc": tokenized_doc,  "ent_id_to_info": ent_id_to_info}


def minimize_partition(args, split, seg_len):
    split_dir = path.join(args.input_dir, SPLIT_TO_DIR[split])
    source_dir = path.join(split_dir, SUBDIR_DICT["source"])
    ann_dir = path.join(split_dir, SUBDIR_DICT["ann"])

    source_files = sorted(glob.glob(path.join(source_dir, "*" + SUBDIR_EXT["source"])))
    # print(source_files, path.join(source_dir, SUBDIR_EXT["source"]))
    doc_ids, ann_files = [], []
    proc_source_files = []
    for source_file in source_files:
        doc_id = path.splitext(path.basename(source_file))[0]
        doc_ids.append(doc_id)
        ann_files.append(path.join(ann_dir, doc_id + SUBDIR_EXT["ann"]))
        proc_source_file = path.join(args.proc_dir, path.basename(source_file) + ".json")
        proc_source_files.append(proc_source_file)

    output_path = path.join(args.output_dir, "{}.{}.jsonlines".format(split, seg_len))
    count = 0
    print("Minimizing {}".format(split))
    with open(output_path, "w") as output_file:
        for doc_name, source_file, proc_source_file, annotation_file in \
                zip(doc_ids, source_files, proc_source_files, ann_files):
            output_dict = tokenize_doc(
                doc_name, source_file, proc_source_file, annotation_file, no_speaker_tags=args.no_speaker_tags)
            document = get_document(doc_name, seg_len, output_dict)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args, seg_len):
    for split in ["dev", "test", "train"]:
        minimize_partition(args, split, seg_len)


if __name__ == "__main__":
    input_dir = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory root")
    parser.add_argument("proc_dir", type=str, help="Directory root of files processed with CoreNLP")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("-no_speaker_tags", action="store_true", default=False, help="Output directory")

    parsed_args = parser.parse_args()

    if not os.path.isdir(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)
    for window_len in [512]:
        minimize_split(parsed_args, window_len)
