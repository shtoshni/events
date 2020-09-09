import torch
import torch.nn as nn
import math

from pytorch_utils.modules import MLP
from red_utils.constants import DOC_TYPE_TO_IDX, ELEM_TYPE_TO_IDX


LOG2 = math.log(2)


class BaseMemory(nn.Module):
    def __init__(self, hsize=300, mlp_size=200, mlp_depth=1,
                 mem_size=None, drop_module=None, emb_size=20, use_srl=False,
                 **kwargs):
        super(BaseMemory, self).__init__()
        # self.query_mlp = query_mlp
        self.hsize = hsize
        self.mem_size = (mem_size if mem_size is not None else hsize)
        # if self.use_srl:
        #     self.mem_size = 2 * self.mem_size
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.mlp_depth = mlp_depth

        self.use_srl = use_srl

        self.drop_module = drop_module

        self.action_str_to_idx = {'c': 0, 'o': 1, 'i': 2, '<s>': 3}
        self.action_idx_to_str = ['c', 'o', 'i']

        self.doc_type_to_idx = DOC_TYPE_TO_IDX

        # CHANGE THIS PART
        self.query_projector = nn.Linear(self.hsize + 4 * self.emb_size, self.mem_size)

        self.mem_coref_mlp = MLP(3 * self.mem_size + 2 * self.emb_size, self.mlp_size, 1,
                                 num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)

        if self.use_srl:
            self.srl_role_mlp = MLP(3 * self.mem_size + 2 * self.emb_size, self.mlp_size, 1,
                                    num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)
            self.srl_coref_mlp = MLP(3 * self.mem_size + 2 * self.emb_size, self.mlp_size, 1,
                                     num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)

        self.ment_type_emb = nn.Embedding(2, self.emb_size)
        self.doc_type_emb = nn.Embedding(3, self.emb_size)
        self.last_action_emb = nn.Embedding(4, self.emb_size)
        self.distance_embeddings = nn.Embedding(11, self.emb_size)
        self.width_embeddings = nn.Embedding(20, self.emb_size)
        self.counter_embeddings = nn.Embedding(11, self.emb_size)

    @staticmethod
    def get_distance_bucket(distances):
        logspace_idx = torch.floor(torch.log(distances.float()) / LOG2).long() + 3
        use_identity = (distances <= 4).long()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    @staticmethod
    def get_counter_bucket(count):
        logspace_idx = torch.floor(torch.log(count.float()) / LOG2).long() + 3
        use_identity = (count <= 4).long()
        combined_idx = use_identity * count + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    @staticmethod
    def get_mention_width_bucket(width):
        if width < 19:
            return width

        return 19

    def get_distance_emb(self, distance):
        distance_tens = self.get_distance_bucket(distance)
        distance_embs = self.distance_embeddings(distance_tens)
        return distance_embs

    def get_counter_emb(self, ent_counter):
        counter_buckets = [self.get_counter_bucket(ent_count) for ent_count in ent_counter]
        counter_tens = torch.tensor(counter_buckets).long().cuda()
        counter_embs = self.counter_embeddings(counter_tens)
        return counter_embs

    def get_last_action_emb(self, action_str):
        action_emb = self.action_str_to_idx[action_str]
        # print(action_str)
        return self.last_action_emb(torch.tensor(action_emb).cuda())

    @staticmethod
    def get_coref_mask(ent_counter, ment_type, cluster_type):
        counter_mask = (ent_counter > 0.0).float().cuda()
        # type_mask = torch.ones_like(counter_mask)
        type_mask = (torch.tensor(ment_type).cuda() == cluster_type).float().cuda()
        # print(type_mask, ent_type, ment_type)
        cell_mask = counter_mask * type_mask
        return cell_mask

    @staticmethod
    def get_srl_mask(ent_counter, ment_type, cluster_type):
        counter_mask = (ent_counter > 0.0).float().cuda()
        # type_mask = torch.ones_like(counter_mask)
        other_type = (1 + ment_type) % 2
        type_mask = (torch.tensor(other_type).cuda() == cluster_type).float().cuda()
        # Reverse mask!
        cell_mask = counter_mask * type_mask
        return cell_mask

    def get_coref_new_log_prob(self, query_vector, ment_type, distance_embs, counter_embs):
        # Repeat the query vector for comparison against all cells
        num_cells = self.mem.shape[0]
        rep_query_vector = query_vector.repeat(num_cells, 1)  # M x H

        # Coref Score
        pair_vec = torch.cat([self.mem, rep_query_vector, self.mem * rep_query_vector,
                              distance_embs, counter_embs], dim=-1)

        pair_score = self.mem_coref_mlp(pair_vec)

        srl_vec, use_srl_mask = None, None
        if self.use_srl:
            if (self.use_srl == 'joint') or ((self.use_srl == 'event') and (ment_type == ELEM_TYPE_TO_IDX['EVENT'])):
                # SRL vec
                srl_vec, use_srl_mask = self.get_srl_role_vec(
                    query_vector, ment_type, distance_embs, counter_embs)
                if use_srl_mask:
                    rep_srl_vec = srl_vec.repeat(num_cells, 1)
                    pair_vec = torch.cat([self.srl_mem, rep_srl_vec, self.srl_mem * rep_srl_vec,
                                          distance_embs, counter_embs], dim=-1)
                    # srl_mask = self.get_srl_mask(self.ent_counter, ment_type, self.cluster_type)
                    srl_score = self.srl_coref_mlp(pair_vec)  # * srl_mask

                    pair_score += srl_score
            else:
                srl_vec = torch.zeros_like(query_vector)

        coref_score = torch.squeeze(pair_score, dim=-1)  # M

        coref_new_mask = torch.cat([self.get_coref_mask(self.ent_counter, ment_type, self.cluster_type),
                                    torch.tensor([1.0]).cuda()], dim=0)
        coref_new_scores = torch.cat(([coref_score, torch.tensor([0.0]).cuda()]), dim=0)
        coref_new_scores = coref_new_scores * coref_new_mask + (1 - coref_new_mask) * (-1e4)

        coref_new_log_prob = torch.nn.functional.log_softmax(coref_new_scores, dim=0)
        return coref_new_scores, coref_new_log_prob, srl_vec, use_srl_mask

    def get_srl_role_vec(self, query_vector, ment_type, distance_embs, counter_embs):
        # Repeat the query vector for comparison against all cells
        num_cells = self.srl_mem.shape[0]
        rep_query_vector = query_vector.repeat(num_cells, 1)  # M x H

        # SRL Score
        pair_vec = torch.cat([self.srl_mem, rep_query_vector, self.srl_mem * rep_query_vector,
                              distance_embs, counter_embs], dim=-1)
        pair_score = self.srl_role_mlp(pair_vec)

        srl_score = torch.squeeze(pair_score, dim=-1)  # M
        # Adding the option for picking None
        srl_no_score = torch.cat(([srl_score, torch.tensor([0.0]).cuda()]), dim=0)

        srl_mask = self.get_srl_mask(self.ent_counter, ment_type, self.cluster_type)
        use_srl_mask = (torch.sum(srl_mask) > 0)
        if use_srl_mask:
            srl_no_mask = torch.cat([srl_mask, torch.tensor([1.0]).cuda()], dim=0)

            # Softmax
            srl_prob = srl_no_mask * torch.nn.functional.softmax(srl_no_score, dim=0)
            srl_prob = srl_prob / (torch.sum(srl_no_mask) + 1e-8)

            # Weighted-avg SRL vector - remove the last term which corresponds to NULL vector
            srl_vec = torch.mv(torch.transpose(self.srl_mem, 1, 0), srl_prob[:-1])
            return srl_vec, use_srl_mask
        else:
            return torch.zeros_like(query_vector), use_srl_mask

    def forward(self, doc_type, mention_emb_list, actions, mentions, teacher_forcing=False):
        pass
