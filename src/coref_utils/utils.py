from collections import defaultdict

def get_mention_to_cluster(clusters, threshold=1):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    mention_to_cluster_dict = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster
    return clusters, mention_to_cluster_dict


def get_mention_to_type_cluster_idx(clusters, threshold=1):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    mention_to_cluster_dict = defaultdict(set)
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            mention_to_cluster_dict[tuple(mention[:2])].add((mention[2], cluster_idx))
    return mention_to_cluster_dict
