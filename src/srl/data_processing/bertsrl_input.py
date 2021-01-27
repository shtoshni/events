from allennlp.predictors.predictor import Predictor
import json
from os import path
import os
from collections import defaultdict
import copy


def get_sentence_idx(doc_offsets, span_boundary):
    for sent_idx, (sentence_start, sentence_end) in enumerate(doc_offsets):
        span_start, span_end = span_boundary
        if span_start >= sentence_start and span_end <= sentence_end:
            return sent_idx

    return None


def process_split(input_dir, output_dir, split="dev"):
    input_file = path.join(input_dir, f"{split}.512.jsonlines")
    output_file = path.join(output_dir, f"{split}.jsonl")

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file) as reader_f, open(output_file, "w") as writer_f:
        for instance_idx, line in enumerate(reader_f):
            # if instance_idx > 0:
            #     break
            instance = json.loads(line.strip())
            doc = []
            doc_offsets = []
            offset = 0
            for sentence in instance["sentences"]:
                doc.extend(sentence)
                doc_offsets.append((offset, offset + len(sentence) - 1))
                offset += len(sentence)

            subtoken_map = instance["subtoken_map"]
            mentions = {}
            for cluster in instance["clusters"]:
                for mention_info in cluster:
                    mentions[mention_info[0]] = subtoken_map[mention_info[1]] - subtoken_map[mention_info[0]] + 1

            for tokenized_sentence in instance["tokenized_sentences"]:
                sentence, offsets = zip(*tokenized_sentence)
                if len(sentence) > 125:
                    # Some sentences are too long!
                    # print("Hello")
                    # print(instance["doc_key"], sentence)
                    # print()
                    continue

                output_dict = {"seq_words": sentence, "BIO": ['O'] * len(sentence), "pred_sense": []}
                for token_idx, offset in enumerate(offsets):
                    if offset[0] in mentions:
                        mention_len = mentions[offset[0]]
                        pred_dict = copy.deepcopy(output_dict)
                        for pred_idx in range(token_idx, token_idx + mention_len):
                            pred_dict["BIO"][pred_idx] = 'B-V'
                        pred_dict['pred_sense'] = [token_idx, sentence[token_idx], f"{sentence[token_idx]}.01", "FAKE"]
                        writer_f.write(json.dumps(pred_dict) + "\n")


def main():
    input_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/cleaned"
    output_dir = "/home/shtoshni/Research/events/proc_data/conll09/bertsrl_input"

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ["dev", "test", "train"]:
        process_split(input_dir, output_dir, split=split)


if __name__ == '__main__':
    main()
