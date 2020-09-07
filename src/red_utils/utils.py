import json
from os import path


def load_data(data_dir, max_segment_len, dataset='red'):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
        with open(jsonl_file) as f:
            split_data = []
            for line in f:
                split_data.append(json.loads(line.strip()))
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    if dataset == 'red':
        assert(len(train_data) == 76)
        assert(len(dev_data) >= 8)
        assert(len(test_data) == 10)

    return train_data, dev_data, test_data


def get_doc_type(example):
    return example["doc_key"].split("/")[0]