from allennlp.predictors.predictor import Predictor
import json
from os import path
import os
from collections import defaultdict
from srl.inference.inference import Inference
from srl.constants import LABELS


def get_sentence_idx(doc_offsets, span_boundary):
    for sent_idx, (sentence_start, sentence_end) in enumerate(doc_offsets):
        span_start, span_end = span_boundary
        if span_start >= sentence_start and span_end <= sentence_end:
            return sent_idx

    return None


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
            doc_offsets = []
            offset = 0
            for sentence in instance["sentences"]:
                doc.extend(sentence)
                doc_offsets.append((offset, offset + len(sentence) - 1))
                offset += len(sentence)

            srl_info_dict = defaultdict(list)

            mentions = set()
            for cluster in instance["clusters"]:
                for mention_info in cluster:
                    mentions.add(tuple(mention_info[:2]))

            for tokenized_sentence in instance["tokenized_sentences"]:
                sentence, offsets = zip(*tokenized_sentence)
                if len(sentence) > 100:
                    # Some sentences are too long!
                    continue

                doc = {"sentence": sentence}
                for token_idx, offset in enumerate(offsets):
                    if tuple(offset) in mentions:
                        doc["predicate"] = token_idx
                        arg_list = srl_model.perform_srl(doc)
                        if len(arg_list):
                            # srl_info = [[] for _ in LABELS]
                            # # Label 0 corresponds to NULL arg. We just instead replace that by predicate.
                            # srl_info[0] = offsets[token_idx]
                            for arg_info in arg_list:
                                t_idx, arg_idx = arg_info[:2]
                                pred_sent_idx = get_sentence_idx(doc_offsets, offset)
                                arg_sent_idx = get_sentence_idx(doc_offsets, offsets[t_idx])

                                assert (pred_sent_idx is not None)
                                assert (arg_sent_idx is not None)
                                assert (pred_sent_idx == arg_sent_idx)

                                srl_info_dict[arg_idx].append((*offset, *offsets[t_idx], pred_sent_idx))

                            # srl_info_list.append(srl_info)

            output_dict = dict(instance)
            output_dict['srl_info'] = srl_info_dict
            writer_f.write(json.dumps(output_dict) + "\n")


def main():
    model_path = "/home/shtoshni/Research/events/models/" \
                 "srl_conll09_mlp_200_model_base_emb_endpoint_type_spanbert_drop_0.0ft/best_models/model.pth"

    srl_model = Inference(model_path)
    input_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/cleaned"
    output_dir = "/home/shtoshni/Research/events/proc_data/kbp_2015/srl"
    for split in ["dev", "test", "train"]:
        process_split(srl_model, input_dir, output_dir, split=split)


if __name__ == '__main__':
    main()
