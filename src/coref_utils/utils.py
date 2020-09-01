from red_utils.constants import  ELEM_TYPE_TO_IDX


def mention_to_cluster(clusters, threshold=1, focus_group='joint'):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    filt_clusters = []
    for cluster in clusters:
        mention = cluster[0]
        if (focus_group == 'entity') and (mention[2] == ELEM_TYPE_TO_IDX['ENTITY']):
            filt_clusters.append(cluster)

        if (focus_group == 'event') and (mention[2] == ELEM_TYPE_TO_IDX['EVENT']):
            filt_clusters.append(cluster)

        if focus_group == 'joint':
            filt_clusters.append(cluster)

    # print(focus_group, len(clusters), len(filt_clusters))

    mention_to_cluster_dict = {}
    for cluster in filt_clusters:
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster

    return filt_clusters, mention_to_cluster_dict
