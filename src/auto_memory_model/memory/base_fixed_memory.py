import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
from auto_memory_model.memory.base_memory import BaseMemory


class BaseFixedMemory(BaseMemory):
    def __init__(self, num_cells=10, **kwargs):
        super(BaseFixedMemory, self).__init__(**kwargs)

        self.num_cells = num_cells

        # Fixed memory cells need to predict fertility of memory and mentions
        self.fert_mlp = MLP(input_size=self.mem_size + 2 * self.emb_size,
                            hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                            bias=True, drop_module=self.drop_module)
        # self.ment_fert_mlp = MLP(input_size=self.mem_size, hidden_size=self.mlp_size,
        #                          output_size=1, num_layers=self.mlp_depth, bias=True,
        #                          drop_module=self.drop_module)
        self.mem, self.srl_mem, self.ent_counter, self.last_mention_idx, self.cluster_type = (None, None, None, None,
                                                                                              None)

    def initialize_memory(self):
        """Initialize the memory to null."""
        self.mem = torch.zeros(self.num_cells, self.mem_size).cuda()
        self.srl_mem = torch.zeros(self.num_cells, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0 for i in range(self.num_cells)]).cuda()
        self.last_mention_idx = torch.tensor([0 for _ in range(self.num_cells)]).cuda()
        self.cluster_type = torch.ones(self.num_cells).cuda() * -1

    def get_overwrite_ign_mask(self, ent_counter):
        last_unused_cell = None
        for cell_idx, cell_count in enumerate(ent_counter.tolist()):
            if int(cell_count) == 0:
                last_unused_cell = cell_idx
                break

        if last_unused_cell is not None:
            score_arr = ([0] * last_unused_cell + [1]
                         + [0] * (self.num_cells - last_unused_cell))
            # Return the overwrite probability vector combined with ignore prob (last entry)
            return torch.tensor(score_arr).cuda().float()
        else:
            return torch.tensor([1] * (1 + self.num_cells)).cuda().float()

    def get_all_mask(self, ent_counter):
        coref_mask = self.get_coref_mask(ent_counter)
        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        return torch.cat([coref_mask, overwrite_ign_mask], dim=0)

    def forward(self, doc_type, mention_emb_list, actions, mentions, teacher_forcing=False):
        pass
