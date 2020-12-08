from collections import defaultdict


def get_ent_info(xml_root):
    """Given the root of elment tree, returns the entity and events."""
    ent_map = {}
    ent_list = []

    for elem in xml_root.iter('entity'):
        span_str = list(elem.iter('span'))[0].text
        span_start, span_end = [int(endpoint) for endpoint in span_str.split(",")]
        elem_id = list(elem.iter('id'))[0].text
        elem_type = list(elem.iter('type'))[0].text

        if elem_type == 'ENTITY' or elem_type == 'EVENT':
            ent_map[elem_id] = (elem_type, (span_start, span_end))
            ent_list.append([(span_start, span_end), elem_id])

    # Sort entity list on the basis of span start index
    ent_list = sorted(ent_list, key=lambda x: x[0][0])

    return ent_map, ent_list


def get_clusters_from_xml(xml_root, ent_map):
    clusters = []
    for elem in xml_root.iter('relation'):
        elem_type = list(elem.iter('type'))[0].text
        if elem_type == 'IDENTICAL':
            # Initiate new cluster
            new_cluster = []
            elem_props = list(elem.iter('properties'))[0]

            for sub_elem in elem_props.iter():
                if (sub_elem.tag == 'FirstInstance') \
                        or (sub_elem.tag == 'Coreferring_String'):
                    ent_id = sub_elem.text
                    assert(ent_id in ent_map)
                    new_cluster.append(ent_id)

            clusters.append(new_cluster)

    return clusters


def get_all_clusters_from_xml(xml_root, ent_map, LEVEL=0):
    cluster_to_type_counter = defaultdict(int)
    mention_to_cluster_info = defaultdict(list)

    for elem in xml_root.iter('relation'):
        elem_type = list(elem.iter('type'))[0].text
        if elem_type == 'IDENTICAL':
            elem_props = list(elem.iter('properties'))[0]
            for sub_elem in elem_props.iter():
                if (sub_elem.tag == 'FirstInstance') \
                        or (sub_elem.tag == 'Coreferring_String'):
                    ent_id = sub_elem.text
                    assert(ent_id in ent_map)
                    mention_to_cluster_info[ent_id].append(
                        ('ID', cluster_to_type_counter[elem_type]))
        elif elem_type == 'BRIDGING' and LEVEL >= 1:
            elem_props = list(elem.iter('properties'))[0]
            for sub_elem in elem_props.iter():
                ent_id = sub_elem.text
                if (sub_elem.tag == 'Argument'):
                    mention_to_cluster_info[ent_id].append(
                        ('BR-arg', cluster_to_type_counter[elem_type]))
                elif (sub_elem.tag == 'Related_to'):
                    mention_to_cluster_info[ent_id].append(
                        ('BR-rel', cluster_to_type_counter[elem_type]))
        elif elem_type == 'WHOLE/PART' and LEVEL >= 2:
            elem_props = list(elem.iter('properties'))[0]
            for sub_elem in elem_props.iter():
                ent_id = sub_elem.text
                if (sub_elem.tag == 'Whole'):
                    mention_to_cluster_info[ent_id].append(
                        ('WP-whole', cluster_to_type_counter[elem_type]))
                elif (sub_elem.tag == 'Part'):
                    mention_to_cluster_info[ent_id].append(
                        ('WP-part', cluster_to_type_counter[elem_type]))
        elif elem_type == 'SET/MEMBER' and LEVEL >= 2:
            elem_props = list(elem.iter('properties'))[0]
            for sub_elem in elem_props.iter():
                ent_id = sub_elem.text
                if (sub_elem.tag == 'Set'):
                    mention_to_cluster_info[ent_id].append(
                        ('SM-set', cluster_to_type_counter[elem_type]))
                elif (sub_elem.tag == 'Member'):
                    mention_to_cluster_info[ent_id].append(
                        ('SM-mem', cluster_to_type_counter[elem_type]))
        elif elem_type == 'APPOSITIVE' and LEVEL >= 1:
            elem_props = list(elem.iter('properties'))[0]
            for sub_elem in elem_props.iter():
                ent_id = sub_elem.text
                if (sub_elem.tag == 'Head'):
                    mention_to_cluster_info[ent_id].append(
                        ('AP-head', cluster_to_type_counter[elem_type]))
                elif (sub_elem.tag == 'Attribute'):
                    mention_to_cluster_info[ent_id].append(
                        ('AP-attr', cluster_to_type_counter[elem_type]))

        cluster_to_type_counter[elem_type] += 1

    return mention_to_cluster_info

