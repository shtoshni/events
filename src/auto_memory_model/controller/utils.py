from auto_memory_model.controller import *


def pick_controller(mem_type='unbounded', dataset='kbp_2015', device='cuda', **kwargs):
    if mem_type == 'unbounded':
        model = UnboundedMemController(dataset=dataset, **kwargs).to(device)
    elif mem_type == 'unbounded_rnn':
        model = UnboundedRNNMemController(dataset=dataset, **kwargs).to(device)
    else:
        raise NotImplementedError(mem_type)

    return model

