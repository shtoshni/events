from os import path
import json


def load_data(data_dir):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = path.join(data_dir, "{}.json".format(split))
        with open(jsonl_file) as f:
            split_data = []
            for line in f:
                split_data.append(json.loads(line.strip()))
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    return train_data, dev_data, test_data
