import torch
from auto_memory_model.memory.base_fixed_memory import BaseMemory
from pytorch_utils.modules import MLP
from data_utils.constants import ELEM_TYPE_TO_IDX


class AlternateMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(AlternateMemory, self).__init__(**kwargs)
        self.srl_coref_mlp = MLP(3 * self.mem_size, self.mlp_size, 1,
                           num_hidden_layers=self.mlp_depth, bias=True, drop_module=self.drop_module)
        self.srl_role_mlp = MLP(3 * self.mem_size, self.mlp_size, 1,
                             num_hidden_layers=self.mlp_depth, bias=True, drop_module=self.drop_module)

        self.mem_vectors = torch.zeros(1, self.mem_size).cuda()
        self.srl_mem = torch.zeros(1, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0.0]).cuda()
        self.last_mention_idx = torch.zeros(1).long().cuda()
        self.cluster_type = torch.tensor([-1]).cuda()

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        self.mem_vectors = torch.zeros(1, self.mem_size).cuda()
        self.srl_mem = torch.zeros(1, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0.0]).cuda()
        self.last_mention_idx = torch.zeros(1).long().cuda()
        self.cluster_type = torch.tensor([-1]).cuda()

    def get_srl_mask(self, ment_type):
        counter_mask = (self.ent_counter > 0.0).float().cuda()
        type_mask = (torch.tensor(ment_type).cuda() == self.cluster_type).float().cuda()
        # Reverse mask!
        cell_mask = counter_mask * (1 - type_mask)
        return cell_mask

    def get_coref_new_scores(self, query_vector, ment_type, ment_score, feature_embs):
        # Repeat the query vector for comparison against all cells

        # Coref Score
        coref_mask = self.get_coref_mask(ment_type)
        use_coref_mask = (torch.sum(coref_mask) > 0)
        coref_score = torch.zeros_like(coref_mask)

        if use_coref_mask:
            indices = torch.squeeze(torch.nonzero(coref_mask, as_tuple=False), dim=-1)
            rel_mems = self.mem_vectors[indices]

            num_cells = rel_mems.shape[0]
            rep_query_vector = query_vector.repeat(num_cells, 1)  # M x H

            pair_vec = torch.cat([rel_mems, rep_query_vector, rel_mems * rep_query_vector,
                                  feature_embs[indices]], dim=-1)
            pair_score = torch.squeeze(self.mem_coref_mlp(pair_vec), dim=-1)
            non_zero_coref_score = pair_score + ment_score  # M

            # for idx in range(indices.shape[0]):
            #     coref_score[indices[idx]] = non_zero_coref_score[idx]
            coref_score[indices] = non_zero_coref_score

        srl_vec = torch.zeros_like(query_vector)
        if ment_type == ELEM_TYPE_TO_IDX['EVENT']:
            # SRL vec
            srl_vec, use_srl_mask = self.get_srl_role_vec(query_vector, ment_type)
            if use_srl_mask:
                # SRL vec is meaningful
                num_cells = self.mem_vectors.shape[0]
                rep_srl_vec = srl_vec.repeat(num_cells, 1)

                # Perform coreference between SRL memory and SRL vec
                srl_pair_vec = torch.cat([self.srl_mem, rep_srl_vec, self.srl_mem * rep_srl_vec], dim=-1)
                srl_mask = self.get_coref_mask(ment_type)
                srl_score = torch.squeeze(self.srl_coref_mlp(srl_pair_vec), dim=-1) * srl_mask
                coref_score += srl_score

        coref_new_mask = torch.cat([self.get_coref_mask(ment_type), torch.tensor([1.0]).cuda()], dim=0)
        coref_new_scores = torch.cat(([coref_score, torch.tensor([0.0]).cuda()]), dim=0)

        coref_new_not_scores = coref_new_scores * coref_new_mask + (1 - coref_new_mask) * (-1e4)
        return coref_new_not_scores, srl_vec

    def get_srl_role_vec(self, query_vector, ment_type):
        # Repeat the query vector for comparison against all cells
        srl_mask = self.get_srl_mask(ment_type)
        use_srl_mask = (torch.sum(srl_mask) > 0)

        if use_srl_mask:
            # SRL Score
            indices = torch.squeeze(torch.nonzero(srl_mask, as_tuple=False), dim=-1)
            ent_vecs = self.mem_vectors[indices]

            rep_query_vector = query_vector.repeat(ent_vecs.shape[0], 1)  # M x H

            pair_vec = torch.cat([ent_vecs, rep_query_vector, ent_vecs * rep_query_vector], dim=-1)
            pair_score = self.srl_role_mlp(pair_vec)

            srl_score = torch.squeeze(pair_score, dim=-1)  # M

            # Adding the option for picking NULL
            srl_no_score = torch.cat(([srl_score, torch.tensor([0.0]).cuda()]), dim=0)

            # Softmax
            srl_prob = torch.nn.functional.softmax(srl_no_score, dim=0)
            srl_prob = srl_prob

            # Weighted-avg SRL vector - remove the last term which corresponds to NULL vector
            srl_vec = torch.mv(torch.transpose(ent_vecs, 1, 0), srl_prob[:-1])
            return srl_vec, use_srl_mask
        else:
            return torch.zeros_like(query_vector), use_srl_mask

    def predict_action(self, query_vector, ment_type, ment_score, feature_embs):
        coref_new_scores, srl_vec = self.get_coref_new_scores(query_vector, ment_type, ment_score, feature_embs)

        # Negate the mention score
        not_a_ment_score = -ment_score

        over_ign_score = torch.cat([torch.tensor([0.0]).cuda(), not_a_ment_score], dim=0).cuda()
        return coref_new_scores, over_ign_score, srl_vec

    def interpret_scores(self, coref_new_scores, overwrite_ign_scores, first_overwrite):
        if first_overwrite:
            num_ents = 0
            num_cells = 1
        else:
            num_ents = coref_new_scores.shape[0] - 1
            num_cells = num_ents

        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < num_cells:
            # Coref
            return pred_max_idx, 'c'
        elif pred_max_idx == num_cells:
            # Overwrite/Invalid mention
            over_max_idx = torch.argmax(overwrite_ign_scores).item()
            if over_max_idx == 0:
                return num_ents, 'o'
            else:
                # Invalid mention
                return -1, 'i'
        else:
            raise NotImplementedError

    def forward(self, mention_emb_list, mention_scores, pred_mentions, gt_actions, metadata, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        self.initialize_memory()

        action_logit_list = []
        action_list = []  # argmax actions
        first_overwrite = True
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        for ment_idx, (ment_emb,  (span_start, span_end, ment_type), ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, pred_mentions, mention_scores, gt_actions)):
            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            feature_embs = self.get_feature_embs(ment_idx, metadata)
            ment_type_emb = self.ment_type_emb(torch.tensor(ment_type).long().cuda())

            if self.use_ment_type:
                query_vector = ment_emb
            else:
                query_vector = self.query_projector(torch.cat([ment_emb, ment_type_emb], dim=0))

            if not (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability
                coref_new_scores, overwrite_ign_scores, srl_vec = self.predict_action(
                    query_vector, ment_type, ment_score, feature_embs)

                pred_cell_idx, pred_action_str = self.interpret_scores(
                    coref_new_scores, overwrite_ign_scores, first_overwrite)
                action_logit_list.append((coref_new_scores, overwrite_ign_scores))

                action_list.append((pred_cell_idx, pred_action_str))
            else:
                continue

            if follow_gt:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

            last_action_str = action_str

            if first_overwrite and action_str == 'o':
                first_overwrite = False
                # We start with a single empty memory cell
                self.mem_vectors = torch.unsqueeze(query_vector, dim=0)
                self.ent_counter = torch.tensor([1.0]).cuda()
                self.last_mention_idx[0] = ment_idx
                self.cluster_type[0] = ment_type
            else:
                num_ents = self.mem_vectors.shape[0]
                # Update the memory
                cell_mask = (torch.arange(0, num_ents) == cell_idx).float().cuda()
                mask = torch.unsqueeze(cell_mask, dim=1)
                mask = mask.repeat(1, self.mem_size)

                # print(cell_idx, action_str, mem_vectors.shape[0])
                if action_str == 'c':
                    self.coref_update(query_vector, cell_idx, mask)
                    self.ent_counter = self.ent_counter + cell_mask
                    self.last_mention_idx[cell_idx] = ment_idx

                    if self.use_ment_type:
                        assert (ment_type == self.cluster_type[cell_idx])

                    if ment_type == ELEM_TYPE_TO_IDX['EVENT']:
                        # Update SRL memory
                        updated_vec = self.get_srl_role_vec(self.mem_vectors[cell_idx], ment_type)[0]
                        self.srl_mem = self.srl_mem * (1 - mask) + mask * torch.unsqueeze(updated_vec, dim=0)

                elif action_str == 'o':
                    # Append the new vector
                    self.mem_vectors = torch.cat([self.mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)
                    self.srl_mem = torch.cat([self.srl_mem, torch.unsqueeze(srl_vec, dim=0)], dim=0)
                    self.ent_counter = torch.cat([self.ent_counter, torch.tensor([1.0]).cuda()], dim=0)
                    self.last_mention_idx = torch.cat([self.last_mention_idx, torch.tensor([ment_idx]).cuda()], dim=0)
                    self.cluster_type = torch.cat([self.cluster_type, torch.tensor([ment_type]).cuda()], dim=0)

        return action_logit_list, action_list
