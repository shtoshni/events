from red_utils.constants import ELEM_TYPE_TO_IDX, IDX_TO_ELEM_TYPE


def get_cluster_type(cluster):
    ment_types = list(zip(*cluster))[2]
    ment_type_frac = sum(ment_types)/len(ment_types)
    if ment_type_frac < 0.5:
        # Majority of spans are entities
        return [IDX_TO_ELEM_TYPE[0]]
    elif ment_type_frac == 0.5:
        # Split between event and entity spans
        return [IDX_TO_ELEM_TYPE[0], IDX_TO_ELEM_TYPE[1]]
    else:
        # Entity spans
        return [IDX_TO_ELEM_TYPE[1]]


def mention_to_cluster(clusters, threshold=1, focus_group='joint'):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    filt_clusters = []
    for cluster in clusters:
        mention_type = get_cluster_type(cluster)
        if (focus_group == 'entity') and ('ENTITY' in mention_type):
            filt_clusters.append(cluster)

        if (focus_group == 'event') and ('EVENT' in mention_type):
            filt_clusters.append(cluster)

        if focus_group == 'joint':
            filt_clusters.append(cluster)

    mention_to_cluster_dict = {}
    for cluster in filt_clusters:
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster

    return filt_clusters, mention_to_cluster_dict


def get_mention_to_cluster_idx(clusters, threshold=1):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    mention_to_cluster_dict = {}
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster_idx
    return mention_to_cluster_dict

