

def get_ordered_mentions(clusters):
    """Order all the mentions in the doc w.r.t. span_start and in case of ties span_end."""
    all_mentions = []
    for cluster in clusters:
        all_mentions.extend(cluster)

    # Span start is the main criteria, and span end is used to break ties
    all_mentions = sorted(all_mentions, key=lambda x: x[0] + 1e-5 * x[1])
    all_mentions = [tuple(mention) for mention in all_mentions]
    return all_mentions


def action_sequences_to_clusters(actions):
    clusters = []
    cell_to_clusters = {}

    for cell_idx, action_type, mention in actions:
        if cell_idx is None:
            continue
        if action_type == 'c':
            cell_to_clusters[cell_idx].append(mention)
        elif action_type == 'o':
            # Overwrite
            if cell_idx in cell_to_clusters:
                # Remove the old cluster and initialize the new
                clusters.append(cell_to_clusters[cell_idx])
            cell_to_clusters[cell_idx] = [mention]

    for cell_idx, cluster in cell_to_clusters.items():
        clusters.append(cluster)

    return clusters


def linearize_actions(actions):
    linearized_actions = []
    for ment_action_list in actions:
        if ment_action_list is None:
            continue
        else:
            for action in ment_action_list:
                linearized_actions.append(action)

    return linearized_actions


def get_mention_to_cluster(clusters):
    mention_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            tuple_ment = tuple(mention)
            mention_to_cluster[tuple_ment] = cluster_idx
    return mention_to_cluster
