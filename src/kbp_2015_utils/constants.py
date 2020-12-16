
SPLIT_TO_DIR = {"train": "mod_training", "dev": "mod_dev", "test": "eval"}
SUBDIR_DICT = {"source": "source", "ann": "hopper"}
SUBDIR_EXT = {"source": ".txt", "ann": ".event_hoppers.xml"}


DOC_TYPES = ['multi_post', 'newswire']
DOC_TYPES_TO_IDX = {doc_type: idx for idx, doc_type in enumerate(DOC_TYPES)}


REALIS_VALS = ['actual', 'generic', 'other']
REALIS_VALS_TO_IDX = {realis_val: idx for idx, realis_val in enumerate(REALIS_VALS)}


EVENT_TYPES = ['business', 'conflict', 'contact', 'justice', 'life', 'manufacture', 'movement',
               'personnel', 'transaction']
EVENT_TYPES_TO_IDX = {event_type: idx for idx, event_type in enumerate(EVENT_TYPES)}


EVENT_SUBTYPES = ['business_declarebankruptcy', 'business_endorg', 'business_mergeorg', 'business_startorg',
                  'conflict_attack', 'conflict_demonstrate', 'contact_broadcast', 'contact_contact',
                  'contact_correspondence', 'contact_meet', 'justice_acquit', 'justice_appeal',
                  'justice_arrestjail', 'justice_chargeindict', 'justice_convict', 'justice_execute',
                  'justice_extradite', 'justice_fine', 'justice_pardon', 'justice_releaseparole',
                  'justice_sentence', 'justice_sue', 'justice_trialhearing', 'life_beborn', 'life_die',
                  'life_divorce', 'life_injure', 'life_marry', 'manufacture_artifact',
                  'movement_transportartifact', 'movement_transportperson', 'personnel_elect',
                  'personnel_endposition', 'personnel_nominate', 'personnel_startposition',
                  'transaction_transaction', 'transaction_transfermoney', 'transaction_transferownership']
EVENT_SUBTYPES_TO_IDX = {event_subtype: idx for idx, event_subtype in enumerate(EVENT_SUBTYPES)}


