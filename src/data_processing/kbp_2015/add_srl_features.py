from allennlp.predictors.predictor import Predictor
import json
from os import path
import os
from collections import OrderedDict


def try_simple_alignment(orig_words, proc_words):
    orig_idx = 0
    mapping = {}
    cur_word = ""
    for proc_idx, proc_word in enumerate(proc_words):
        if (proc_word == orig_words[orig_idx]) or ((cur_word + proc_word) == orig_words[orig_idx]):
            mapping[proc_idx] = orig_idx
            orig_idx += 1
            cur_word = ""
        elif proc_word in orig_words[orig_idx]:
            cur_word += proc_word
            mapping[proc_idx] = orig_idx
        else:
            raise ValueError(proc_word, orig_words[orig_idx], orig_words, proc_words)

    assert (orig_idx == len(orig_words))
    return mapping


def parse_output(output_dict):
    tag_dict_list = []
    for verb_dict in output_dict['verbs']:
        tags = verb_dict['tags']
        tag_to_idx = {}
        for idx, tag in enumerate(tags):
            if tag == 'O':
                continue
            else:
                begin_or_in, arg = tag[:1], tag[2:]

                if begin_or_in == 'B':
                    tag_to_idx[arg] = [idx]
                else:
                    try:
                        tag_to_idx[arg].append(idx)
                    except KeyError:
                        # The model can sometimes directly output I-ARGX instead of starting with B-ARGX
                        # Sample Sentence - "Continental Airlines Board discusses merger with United XXXXXXXX :
                        # Jump Headline Goes Herey and Herey"
                        pass

        if 'V' in tag_to_idx:
            tag_dict_list.append(tag_to_idx)
        else:
            print(tag_to_idx)
            print()

    return tag_dict_list


def process_split(srl_model, input_dir, output_dir, split="dev"):
    input_file = path.join(input_dir, f"{split}.512.jsonlines")
    output_file = path.join(output_dir, f"{split}.512.jsonlines")

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file) as reader_f, open(output_file, "w") as writer_f:
        for instance_idx, line in enumerate(reader_f):
            # if instance_idx < 91:
            #     continue
            instance = json.loads(line.strip())
            doc = []
            for sentence in instance["sentences"]:
                doc.extend(sentence)

            srl_info_list = []
            batch_json_list = []
            batch_sentences = []
            batch_offsets = []
            for tokenized_sentence in instance["tokenized_sentences"]:
                sentence, offsets = zip(*tokenized_sentence)
                if len(sentence) > 100:
                    # Some sentences are too long!
                    continue

                sentence_str = " ".join(sentence)
                batch_json_list.append({"sentence": sentence_str})
                batch_sentences.append(sentence)
                batch_offsets.append(offsets)

            pred_list = srl_model.predict_batch_json(batch_json_list)

            for sentence, offsets, pred in zip(batch_sentences, batch_offsets, pred_list):
                if len(pred['verbs']):
                    try:
                        if len(sentence) != len(pred['words']):
                            mapping = try_simple_alignment(sentence, pred['words'])
                        else:
                            mapping = {idx: idx for idx in range(len(sentence))}

                        tag_dict_list = parse_output(pred)
                        for tag_dict in tag_dict_list:
                            verb = tag_dict['V']
                            arg0 = tag_dict.get('ARG0', [])
                            arg1 = tag_dict.get('ARG1', [])
                            loc = tag_dict.get('ARGM-LOC', [])
                            tmp = tag_dict.get('ARGM-TMP', [])

                            srl_info = [verb, arg0, arg1, loc, tmp]

                            mapped_srl_info = []
                            for idx_list in srl_info:
                                span_boundary = []
                                if len(idx_list):
                                    try:
                                        span_boundary = (
                                            offsets[mapping[idx_list[0]]][0], offsets[mapping[idx_list[-1]]][1])
                                    except KeyError:
                                        print("Noooooo")
                                        import sys
                                        sys.exit()

                                mapped_srl_info.append(span_boundary)

                            srl_info_list.append(mapped_srl_info)

                    except AssertionError:
                        print(sentence, pred['words'])
                        break

            output_dict = dict(instance)
            output_dict['srl_info'] = srl_info_list
            writer_f.write(json.dumps(output_dict) + "\n")


def main():
    srl_model = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz", cuda_device=0)

    input_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/cleaned"
    output_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/srl"
    for split in ["dev", "test", "train"]:
        process_split(srl_model, input_dir, output_dir, split=split)


if __name__=='__main__':
    main()