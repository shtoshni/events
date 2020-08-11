import torch
import torch.nn as nn

from pytorch_utils.modules import MLP
from auto_memory_model.memory.base_fixed_memory import BaseMemory
import math


class UnboundedMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemory, self).__init__(**kwargs)

        # self.query_projector = nn.Linear(self.hsize + self.mem_size + 4 * self.emb_size, self.mem_size)
        # self.mem_key_transform = nn.Linear(self.mem_size, self.mem_size)
        # self.mem_val_transform = nn.Linear(self.mem_size, self.mem_size)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        mem = torch.zeros(1, self.mem_size).cuda()
        ent_counter = torch.tensor([0]).cuda()
        last_mention_idx = torch.zeros(1).long().cuda()
        ent_type = torch.tensor([-1]).cuda()
        return mem, ent_counter, last_mention_idx, ent_type

    def predict_action(self, query_vector, ment_type, mem_vectors, ent_type, last_ment_vectors,
                       ment_idx, ent_counter, last_mention_idx):
        distance_embs = self.get_distance_emb(ment_idx - last_mention_idx)
        counter_embs = self.get_counter_emb(ent_counter)

        coref_new_scores, coref_new_log_prob = self.get_coref_new_log_prob(
            query_vector, ment_type, mem_vectors, ent_type, last_ment_vectors,
            ent_counter, distance_embs, counter_embs)

        return coref_new_scores

    def get_memory_context(self, query_vector, mem):
        key_mat = self.mem_key_transform(mem)
        val_mat = self.mem_val_transform(mem)

        const = math.sqrt(val_mat.shape[1])
        sim_scores = torch.nn.functional.softmax(torch.mv(key_mat, query_vector)/const, dim=0)
        context_vec = torch.mv(torch.transpose(val_mat, 1, 0), sim_scores)
        return context_vec

    def forward(self, doc_type, mention_emb_list, actions, mentions,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx, ent_type = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)

        if self.entity_rep == 'lstm':
            cell_vectors = torch.zeros_like(mem_vectors)

        action_logit_list = []
        action_list = []  # argmax actions

        span_type_logit_list = []
        pred_span_type_list = []
        gt_span_type_list = []
        spans_seen = set()

        action_str = '<s>'
        ment_idx = 0

        for span_idx, (ment_emb, (span_start, span_end, _, gt_span_type)) in \
                enumerate(zip(mention_emb_list, mentions)):
            span_endpoints = (span_start, span_end)
            if span_endpoints in spans_seen:
                # print("Hello", span_endpoints)
                continue
            else:
                spans_seen.add(span_endpoints)

            # Doc type embedding
            doc_type_idx = self.doc_type_to_idx[doc_type]
            doc_type_emb = self.doc_type_emb(torch.tensor(doc_type_idx).long().cuda())

            # Embed the width embedding
            width_bucket = self.get_mention_width_bucket(span_end - span_start)
            width_embedding = self.width_embeddings(torch.tensor(width_bucket).long().cuda())

            # Last action embedding
            last_action_emb = self.get_last_action_emb(action_str)

            query_vector = self.query_projector(
                torch.cat([ment_emb, doc_type_emb, width_embedding, last_action_emb], dim=0))

            # Context vector
            # context_vec = self.get_memory_context(ment_emb, mem_vectors)
            # span_type_logit = self.span_type_mlp(
            #     torch.cat([query_vector, context_vec, doc_type_emb, width_embedding], dim=0))

            span_type_logit = self.span_type_mlp(
                torch.cat([query_vector, doc_type_emb, width_embedding], dim=0))
            span_type_logit_list.append(span_type_logit)
            pred_span_type = torch.argmax(span_type_logit).item()
            pred_span_type_list.append((span_endpoints, pred_span_type))
            gt_span_type_list.append((span_endpoints, gt_span_type))

            if self.training or teacher_forcing:
                span_type = gt_span_type
            else:
                span_type = pred_span_type
                # span_type = gt_span_type

            if span_type < 2:
                ment_types = [span_type]
            else:
                # The span refers to both event and entity
                ment_types = [0, 1]

            for ment_type in ment_types:

                # Embed the mention type
                # ment_type_emb = self.ment_type_emb(torch.tensor(ment_type).long().cuda())

                coref_new_scores = self.predict_action(
                    query_vector, ment_type, mem_vectors, ent_type, last_ment_vectors,
                    ment_idx, ent_counter, last_mention_idx)

                action_logit_list.append(coref_new_scores)

                if ment_idx == 0:
                    # We start with a single empty memory cell
                    mem_vectors = torch.unsqueeze(query_vector, dim=0)
                    last_ment_vectors = torch.unsqueeze(query_vector, dim=0)
                    ent_counter = torch.tensor([1.0]).cuda()
                    last_mention_idx[0] = 0
                    ent_type = torch.tensor([ment_type]).cuda()

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
                        cell_idx, action_str = actions[ment_idx]
                    else:
                        # Inference time
                        action_str = pred_action_str
                        cell_idx = pred_cell_idx
                        # cell_idx, action_str = actions[ment_idx]

                    # Update the memory
                    rep_query_vector = query_vector.repeat(num_ents, 1)  # M x H
                    cell_mask = (torch.arange(0, num_ents) == cell_idx).float().cuda()
                    mask = torch.unsqueeze(cell_mask, dim=1)
                    mask = mask.repeat(1, self.mem_size)

                    # print(cell_idx, action_str, mem_vectors.shape[0])
                    if action_str == 'c':
                        # Update memory vector corresponding to cell_idx
                        if self.entity_rep == 'lstm':
                            cand_vec, cand_cell_vec = self.mem_rnn(
                                    rep_query_vector, (mem_vectors, cell_vectors))
                            cell_vectors = cell_vectors * (1 - mask) + mask * cand_cell_vec
                        elif self.entity_rep == 'gru':
                            cand_vec = self.mem_rnn(rep_query_vector, mem_vectors)
                            mem_vectors = mem_vectors * (1 - mask) + mask * cand_vec
                        elif self.entity_rep == 'max':
                            # Max pool coref operation
                            max_pool_vec = torch.max(
                                torch.stack([mem_vectors, rep_query_vector], dim=0), dim=0)[0]
                            mem_vectors = mem_vectors * (1 - mask) + mask * max_pool_vec
                        elif self.entity_rep == 'avg':
                            total_counts = torch.unsqueeze((ent_counter + 1).float(), dim=1)
                            pool_vec_num = (mem_vectors * torch.unsqueeze(ent_counter, dim=1)
                                            + rep_query_vector)
                            avg_pool_vec = pool_vec_num/total_counts
                            mem_vectors = mem_vectors * (1 - mask) + mask * avg_pool_vec

                        # Update last mention vector
                        last_ment_vectors = last_ment_vectors * (1 - mask) + mask * rep_query_vector
                        ent_counter = ent_counter + cell_mask
                        last_mention_idx[cell_idx] = ment_idx

                        # assert(ment_type == ent_type[cell_idx])
                    elif action_str == 'o':
                        # Append the new vector
                        mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)
                        # Update last mention vector
                        last_ment_vectors = torch.cat([last_ment_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)

                        ent_counter = torch.cat([ent_counter, torch.tensor([1.0]).cuda()], dim=0)
                        last_mention_idx = torch.cat([last_mention_idx, torch.tensor([ment_idx]).cuda()], dim=0)
                        ent_type = torch.cat([ent_type, torch.tensor([ment_type]).cuda()], dim=0)

                ment_idx += 1

        # print(ment_idx)
        return (action_logit_list, action_list, span_type_logit_list, pred_span_type_list, gt_span_type_list)
