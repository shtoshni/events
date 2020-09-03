import torch
from auto_memory_model.memory.base_fixed_memory import BaseFixedMemory


class LearnedFixedMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)

    def predict_action(self, query_vector, ment_type, ment_idx):
        distance_embs = self.get_distance_emb(ment_idx - self.last_mention_idx)
        counter_embs = self.get_counter_emb(self.ent_counter)

        coref_new_scores, coref_new_log_prob, srl_vec, use_srl_mask = self.get_coref_new_log_prob(
            query_vector, ment_type, distance_embs, counter_embs)
        # Fertility Score
        # Memory + Mention fertility input
        mem_fert_input = torch.cat([self.mem, distance_embs, counter_embs], dim=-1)
        # Mention fertility input
        ment_distance_emb = torch.squeeze(self.distance_embeddings(torch.tensor([0]).cuda()), dim=0)
        ment_counter_emb = torch.squeeze(self.counter_embeddings(torch.tensor([0]).cuda()), dim=0)
        ment_fert_input = torch.unsqueeze(
            torch.cat([query_vector, ment_distance_emb, ment_counter_emb], dim=0), dim=0)
        # Fertility scores
        fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)
        fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)

        overwrite_ign_mask = self.get_overwrite_ign_mask(self.ent_counter)
        overwrite_ign_scores = fert_scores * overwrite_ign_mask + (1 - overwrite_ign_mask) * (-1e4)
        overwrite_ign_log_prob = torch.nn.functional.log_softmax(overwrite_ign_scores, dim=0)

        norm_overwrite_ign_log_prob = (coref_new_log_prob[self.num_cells] + overwrite_ign_log_prob)
        all_log_prob = torch.cat([coref_new_log_prob[:self.num_cells],
                                  norm_overwrite_ign_log_prob], dim=0)
        return all_log_prob, coref_new_scores, overwrite_ign_scores, srl_vec, use_srl_mask

    def forward(self, doc_type, mention_emb_list, actions, mentions,
                teacher_forcing=False):
        # Initialize memory
        self.initialize_memory()

        action_logit_list = []
        action_list = []  # argmax actions
        action_str = '<s>'

        for ment_idx, (ment_emb, (span_start, span_end, ment_type), (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mentions, actions)):
            # Doc type embedding
            doc_type_idx = self.doc_type_to_idx[doc_type]
            doc_type_emb = self.doc_type_emb(torch.tensor(doc_type_idx).long().cuda())
            # Embed the mention type
            ment_type_emb = self.ment_type_emb(torch.tensor(ment_type).long().cuda())
            # Embed the width embedding
            width_bucket = self.get_mention_width_bucket(span_end - span_start)
            width_embedding = self.width_embeddings(torch.tensor(width_bucket).long().cuda())
            # Last action embedding
            last_action_emb = self.get_last_action_emb(action_str)

            query_vector = self.query_projector(
                torch.cat([ment_emb, doc_type_emb, ment_type_emb,
                           width_embedding, last_action_emb], dim=0))

            all_log_probs, coref_new_scores, overwrite_ign_scores, srl_vec, use_srl_mask = self.predict_action(
                query_vector, ment_type, ment_idx)

            action_logit_list.append((coref_new_scores, overwrite_ign_scores))

            pred_max_idx = torch.argmax(all_log_probs).item()
            pred_cell_idx = pred_max_idx % self.num_cells
            pred_action_idx = pred_max_idx // self.num_cells
            pred_action_str = self.action_idx_to_str[pred_action_idx]
            # During training this records the next actions  - during testing it records the
            # predicted sequence of actions
            action_list.append((pred_cell_idx, pred_action_str))

            if self.training or teacher_forcing:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

            # Update the memory
            rep_query_vector = query_vector.repeat(self.num_cells, 1)  # M x H
            cell_mask = (torch.arange(0, self.num_cells) == cell_idx).float().cuda()
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            if action_str == 'c':
                # Update memory vector corresponding to cell_idx
                total_counts = torch.unsqueeze((self.ent_counter + 1).float(), dim=1)
                pool_vec_num = (self.mem * torch.unsqueeze(self.ent_counter, dim=1)
                                + rep_query_vector)
                avg_pool_vec = pool_vec_num/total_counts
                self.mem = self.mem * (1 - mask) + mask * avg_pool_vec
                if self.use_srl:
                    pool_vec_num = (self.srl_mem * torch.unsqueeze(self.ent_counter, dim=1)
                                    + srl_vec)
                    avg_pool_vec = pool_vec_num / total_counts
                    self.srl_mem = self.srl_mem * (1 - mask) + mask * avg_pool_vec

                # Update last mention vector
                self.ent_counter = self.ent_counter + cell_mask
                self.last_mention_idx[cell_idx] = ment_idx

                assert (ment_type == self.cluster_type[cell_idx])
            elif action_str == 'o':
                # Replace the cell content
                self.mem = self.mem * (1 - mask) + mask * rep_query_vector
                if self.use_srl:
                    self.srl_mem = self.srl_mem * (1 - mask) + mask * srl_vec

                self.ent_counter = self.ent_counter * (1 - cell_mask) + cell_mask
                self.last_mention_idx[cell_idx] = ment_idx
                self.cluster_type[cell_idx] = ment_type

        return action_logit_list, action_list
