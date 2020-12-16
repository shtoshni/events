from kbp_2015_utils.constants import EVENT_TYPES_TO_IDX, EVENT_SUBTYPES


def get_event_type(subevent_val):
    event_type = EVENT_SUBTYPES[subevent_val].split("_")[0]
    return EVENT_TYPES_TO_IDX[event_type]