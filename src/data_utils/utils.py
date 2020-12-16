import json
from os import path


def load_data(data_dir, max_segment_len, dataset='kbp_2015'):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
        with open(jsonl_file) as f:
            split_data = []
            for line in f:
                split_data.append(json.loads(line.strip()))
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    if dataset == 'kbp_2015':
        assert(len(train_data) == 128)
        assert(len(dev_data) >= 30)
        assert(len(test_data) == 202)

    return train_data, dev_data, test_data


def get_clusters(orig_clusters, key="subtype_val"):
    clusters = []

    for orig_cluster in orig_clusters:
        cluster = []
        for (span_start, span_end, mention_info) in orig_cluster:
            cluster.append((span_start, span_end, mention_info[key]))
        clusters.append(cluster)

    return clusters
