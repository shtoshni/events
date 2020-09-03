import torch
from auto_memory_model.memory.base_fixed_memory import BaseMemory


class UnboundedMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemory, self).__init__(**kwargs)
        self.mem, self.srl_mem, self.ent_counter, self.last_mention_idx, self.cluster_type = (None, None, None,
                                                                                              None, None)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        self.mem = torch.zeros(1, self.mem_size).cuda()
        self.srl_mem = torch.zeros(1, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0]).cuda()
        self.last_mention_idx = torch.zeros(1).long().cuda()
        self.cluster_type = torch.tensor([-1]).cuda()
        # return mem, ent_counter, last_mention_idx, cluster_type

    def predict_action(self, query_vector, ment_type, ment_idx):
        distance_embs = self.get_distance_emb(ment_idx - self.last_mention_idx)
        counter_embs = self.get_counter_emb(self.ent_counter)

        coref_new_scores, _, srl_vec, use_srl_mask = self.get_coref_new_log_prob(
            query_vector, ment_type, distance_embs, counter_embs)

        return coref_new_scores, srl_vec

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

            coref_new_scores, srl_vec = self.predict_action(
                query_vector, ment_type, ment_idx)

            action_logit_list.append(coref_new_scores)

            if ment_idx == 0:
                # We start with a single empty memory cell
                self.mem = torch.unsqueeze(query_vector, dim=0)
                if self.use_srl:
                    self.srl_mem = torch.unsqueeze(srl_vec, dim=0)
                self.ent_counter = torch.tensor([1.0]).cuda()
                self.last_mention_idx[0] = 0
                self.cluster_type = torch.tensor([ment_type]).cuda()
                action_list.append((0, 'o'))
            else:
                pred_max_idx = torch.argmax(coref_new_scores).item()
                num_ents = coref_new_scores.shape[0] - 1

                if pred_max_idx == num_ents:
                    pred_action_str = 'o'
                    pred_cell_idx = num_ents
                else:
                    pred_action_str = 'c'
                    pred_cell_idx = pred_max_idx

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
                rep_query_vector = query_vector.repeat(num_ents, 1)  # M x H
                cell_mask = (torch.arange(0, num_ents) == cell_idx).float().cuda()
                mask = torch.unsqueeze(cell_mask, dim=1)
                mask = mask.repeat(1, self.mem_size)

                # print(cell_idx, action_str, mem.shape[0])
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

                    self.ent_counter = self.ent_counter + cell_mask
                    self.last_mention_idx[cell_idx] = ment_idx

                    assert (ment_type == self.cluster_type[cell_idx])
                elif action_str == 'o':
                    # Append the new vector
                    self.mem = torch.cat([self.mem, torch.unsqueeze(query_vector, dim=0)], dim=0)
                    if self.use_srl:
                        self.srl_mem = torch.cat([self.srl_mem, torch.unsqueeze(srl_vec, dim=0)], dim=0)
                    self.ent_counter = torch.cat([self.ent_counter, torch.tensor([1.0]).cuda()], dim=0)
                    self.last_mention_idx = torch.cat([self.last_mention_idx, torch.tensor([ment_idx]).cuda()], dim=0)
                    self.cluster_type = torch.cat([self.cluster_type, torch.tensor([ment_type]).cuda()], dim=0)

        return action_logit_list, action_list
