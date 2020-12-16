
def get_mention_to_cluster(clusters, threshold=1):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    mention_to_cluster_dict = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster
    return clusters, mention_to_cluster_dict


def get_mention_to_cluster_idx(clusters, threshold=1):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    mention_to_cluster_dict = {}
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster_idx
    return mention_to_cluster_dict

