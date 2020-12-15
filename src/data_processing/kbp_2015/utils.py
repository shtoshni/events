import xml.etree.ElementTree as ET
from collections import defaultdict
from kbp_2015_utils.constants import REALIS_VALS_TO_IDX, EVENT_TYPES_TO_IDX, EVENT_SUBTYPES_TO_IDX, DOC_TYPES_TO_IDX


def parse_ann_file(ann_file):
    tree = ET.parse(ann_file)
    xml_root = tree.getroot()

    doc_type = DOC_TYPES_TO_IDX[xml_root.attrib['source_type']]

    mention_list = []
    clusters = []

    unique_spans = defaultdict(set)
    for coref_elem in xml_root.iter('hopper'):
        cur_cluster = []
        for event_elem in coref_elem.iter('event_mention'):
            event_info = {}

            event_id = event_elem.attrib['id']
            trigger_elem = event_elem.find('trigger')

            span_start = int(trigger_elem.attrib['offset'])
            span_end = span_start + int(trigger_elem.attrib['length'])
            subtype = f"{event_elem.attrib['type']}_{event_elem.attrib['subtype']}"

            span_info = (span_start, span_end, subtype)
            if span_info in unique_spans:
                print(f"Detected a duplicate span in {ann_file}: "
                      f"Event ID {event_id} is duplicate with {unique_spans[span_info]}")
                print("DUPLICATE SPANS ARE NOT ADDED TO PROCESSED DOC")
                continue
            else:
                unique_spans[span_info] = event_id

            event_info["id"] = event_id

            event_info["type"] = event_elem.attrib['type']
            event_info["type_val"] = EVENT_TYPES_TO_IDX[event_info["type"]]

            event_info["subtype"] = subtype
            event_info["subtype_val"] = EVENT_SUBTYPES_TO_IDX[event_info["subtype"]]

            event_info["realis"] = event_elem.attrib['realis']
            event_info["realis_val"] = REALIS_VALS_TO_IDX[event_info["realis"]]

            event_phrase = trigger_elem.text
            event_info["text"] = event_phrase

            mention_list.append((span_start, span_end, event_info))

            cur_cluster.append(event_id)

        if cur_cluster:
            clusters.append(cur_cluster)

    return doc_type, mention_list, clusters
