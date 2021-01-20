import json
import argparse
from os import path
from transformers import BertTokenizer
from srl.constants import LABELS_TO_IDX
import copy


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenize_sentence(sentence, tokenizer):
    tokenized_sentence = []
    subtoken_map = []

    for idx, word in enumerate(sentence):
        tokens = tokenizer.tokenize(word)
        if len(tokens) == 0:
            continue

        subtoken_map.append((len(tokenized_sentence), len(tokenized_sentence) + len(tokens) - 1))
        tokenized_sentence.extend(tokens)

    return tokenized_sentence, subtoken_map


def convert_sent_dict(sentence):
    sentences = [line[1] for line in sentence]
    # print(sentences)
    tokenized_sentence, subtoken_map = tokenize_sentence(sentences, tokenizer=bert_tokenizer)

    common_dict = {
        'sentences': [tokenized_sentence],
        'start_idx': [start_idx for start_idx, end_idx in subtoken_map],
        'end_idx': [end_idx for start_idx, end_idx in subtoken_map],
        'predicate': None,
        'args': []
    }

    srl_outputs = []

    predicate_idx = 0
    for idx in range(len(sentence)):
        if sentence[idx][12] == 'Y':  # is predicate
            output_dict = copy.deepcopy(common_dict)
            output_dict['predicate'] = idx
            for jdx in range(len(sentence)):
                arg_label = sentence[jdx][14 + predicate_idx]
                if arg_label in LABELS_TO_IDX:
                    output_dict['args'].append([jdx, LABELS_TO_IDX[arg_label], arg_label,
                                                    sentence[idx][1], sentence[jdx][1]])

            srl_outputs.append(output_dict)
            predicate_idx += 1

    return srl_outputs


def conll09_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.readlines()

    sentences = []
    sent = []
    for line in data:
        if len(line.strip()) > 0:
            sent.append(line.strip().split('\t'))
        else:
            if len(sent) > 0:
                sentences.append(sent)
                sent = []
                # break

    if len(sent) > 0:
        sentences.append(sent)

    with open(output_file, 'w') as f:
        # print(sentences, len(sentences))
        for idx in range(len(sentences)):
            srl_output = convert_sent_dict(sentences[idx])
            for json_data in srl_output:
                f.write(json.dumps(json_data) + '\n')


def process_split(input_dir, output_dir, split='train'):
    split_to_filename = {
        "train": "CoNLL2009-ST-English-train.txt",
        "dev": "CoNLL2009-ST-English-development.txt",
        "test": "CoNLL2009-ST-evaluation-English.txt",
    }

    input_file = path.join(input_dir, split_to_filename[split])
    output_file = path.join(output_dir, f"{split}.json")

    conll09_to_json(input_file, output_file)


def main(args):
    if not path.exists(args.output_dir):
        import os
        os.makedirs(args.output_dir)
    for split in ['train', 'dev', 'test']:
        process_split(args.input_dir, args.output_dir, split=split)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory root")
    parser.add_argument("output_dir", type=str, help="Output directory")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
