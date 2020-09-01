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
            ent_list.append([(span_start, span_end), elem_type, elem_id])

    # Sort entity list on the basis of span start index
    ent_list = sorted(ent_list, key=lambda x: x[0][0])

    return ent_map, ent_list


def get_all_ent_info(xml_root):
    """Given the root of elment tree, returns the entity, events, and some other tags."""
    ent_map = {}
    ent_list = []

    for elem in xml_root.iter('entity'):
        span_str = list(elem.iter('span'))[0].text
        span_start, span_end = [int(endpoint) for endpoint in span_str.split(",")]
        elem_id = list(elem.iter('id'))[0].text
        elem_type = list(elem.iter('type'))[0].text

        if elem_type == 'ENTITY' or elem_type == 'EVENT' or elem_type == 'TIMEX3' or elem_type == 'DUPLICATE':
            ent_map[elem_id] = (elem_type, (span_start, span_end))
            ent_list.append([(span_start, span_end), elem_type, elem_id])

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
