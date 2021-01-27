import json
from os import path
import os
from collections import defaultdict


TAG_TO_IDX = {'A0': 0, 'A1': 1, 'A2': 2, 'AM-TMP': 3, 'AM-LOC': 4,
              'AM-MOD': 5, 'AM-ADV': 5, 'AM-MNR': 5}


def get_sentence_idx(doc_offsets, span_boundary):
    for sent_idx, (sentence_start, sentence_end) in enumerate(doc_offsets):
        span_start, span_end = span_boundary
        if span_start >= sentence_start and span_end <= sentence_end:
            return sent_idx

    return None


def process_split(input_dir, output_dir, model_output_dir, split="dev"):
    input_file = path.join(input_dir, f"{split}.512.jsonlines")
    output_file = path.join(output_dir, f"{split}.512.jsonlines")
    model_output_file = path.join(model_output_dir, f"{split}.jsonl")

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    with open(model_output_file) as f:
        model_output_dict = json.loads(f.read().strip())

    with open(input_file) as reader_f, open(output_file, "w") as writer_f:
        total_mentions = 0
        mention_coverage = 0
        tag_count = defaultdict(int)
        for instance_idx, line in enumerate(reader_f):
            # if instance_idx > 1:
            #     continue
            instance = json.loads(line.strip())
            doc = []
            doc_offsets = []
            offset = 0
            for sentence in instance["sentences"]:
                doc.extend(sentence)
                doc_offsets.append((offset, offset + len(sentence) - 1))
                offset += len(sentence)

            srl_info_dict = defaultdict(list)

            mentions = {}
            for cluster in instance["clusters"]:
                for mention_info in cluster:
                    mentions[mention_info[0]] = [mention_info[0], mention_info[1]]
                    total_mentions += 1
            for tokenized_sentence in instance["tokenized_sentences"]:
                sentence, offsets = zip(*tokenized_sentence)
                sentence_str = " ".join(sentence)
                for token_idx, offset in enumerate(offsets):
                    if offset[0] in mentions:
                        pred_sent_str = f'{sentence_str}\t{token_idx}'
                        if pred_sent_str in model_output_dict:
                            tag_dict = model_output_dict[pred_sent_str]['tag_dict']
                            arg_list = []
                            for tag in tag_dict:
                                if tag in TAG_TO_IDX:
                                    # In rare cases there can be multiple args due to BIO mapping, we just pick up
                                    # the first one
                                    arg_idx = tag_dict[tag][0]
                                    arg_list.append((arg_idx, TAG_TO_IDX[tag]))

                            # print(arg_list, tag_dict)
                            # Check if we have any tags of interest
                            if len(arg_list):
                                for arg_idx, tag_idx in arg_list:
                                    pred_sent_idx = get_sentence_idx(doc_offsets, offset)
                                    arg_sent_idx = get_sentence_idx(doc_offsets, offsets[arg_idx])

                                    assert (pred_sent_idx is not None)
                                    assert (arg_sent_idx is not None)
                                    assert (pred_sent_idx == arg_sent_idx)

                                    tag_count[tag_idx] += 1
                                    srl_info_dict[tag_idx].append((*offset, *offsets[arg_idx], pred_sent_idx))
                                mention_coverage += 1

                            # if len(arg_list) == 0:
                            #     print(arg_list, tag_dict)

            output_dict = dict(instance)
            output_dict['srl_info'] = srl_info_dict
            writer_f.write(json.dumps(output_dict) + "\n")

        print("Coverage:", round(mention_coverage/total_mentions, 3))
        print(mention_coverage, total_mentions)
        print(tag_count)


def main():
    input_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/cleaned"
    model_output_dir = "/home/shtoshni/Research/events/proc_data/conll09/bertsrl_output_proc/"
    output_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/bertsrl"
    for split in ["dev", "test", "train"]:
        process_split(input_dir, output_dir, model_output_dir, split=split)


if __name__ == '__main__':
    main()
