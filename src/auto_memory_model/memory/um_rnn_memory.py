import torch
import torch.nn as nn
from auto_memory_model.memory.base_memory import BaseMemory
from kbp_2015_utils.constants import EVENT_SUBTYPES
from kbp_2015_utils.utils import get_event_type
from pytorch_utils.modules import MLP


class UnboundedRNNMemory(BaseMemory):
    def __init__(self, rnn_size=50, **kwargs):
        super(UnboundedRNNMemory, self).__init__(**kwargs)
        if self.use_mem_context:
            vec_size = self.mem_size
        else:
            vec_size = 0

        if self.use_mem_context:
            self.mem_sim_mlp = MLP(self.mem_size + self.hsize + self.num_feats * self.emb_size, self.mlp_size, 1,
                                   num_hidden_layers=self.mlp_depth, bias=True, drop_module=self.drop_module)

        self.rnn_size = rnn_size
        self.proj_layer = MLP(vec_size + self.hsize + 2 * self.emb_size + 1, self.mlp_size, rnn_size)
        self.event_subtype_rnn = torch.nn.GRUCell(hidden_size=rnn_size, input_size=self.emb_size)
        self.event_subtype_proj = nn.Linear(rnn_size, len(EVENT_SUBTYPES) + 1)
        # self.event_subtype_repr = nn.Embedding(len(EVENT_SUBTYPES), self.emb_size)

        self.sim_softmax = nn.Softmax(dim=0)
        self.eos_event_id = len(EVENT_SUBTYPES)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        self.mem_vectors = torch.zeros(1, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0.0]).cuda()
        self.last_mention_idx = torch.zeros(1).long().cuda()
        self.last_sent_idx = torch.zeros(1).long().cuda()
        self.cluster_type = torch.tensor([-1]).cuda()
        self.last_mention_boundary = []

    def get_mem_context(self, ment_emb, feat_embs):
        num_cells = self.mem_vectors.shape[0]
        rep_ment_emb = ment_emb.repeat(num_cells, 1)  # M x H

        # Pair vec
        pair_vec = torch.cat([self.mem_vectors, rep_ment_emb, feat_embs], dim=-1)
        pair_score = torch.squeeze(self.mem_sim_mlp(pair_vec), dim=-1)

        pair_sim = torch.unsqueeze(self.sim_softmax(pair_score), dim=1)
        sim_vec = torch.sum(pair_sim * self.mem_vectors, dim=0)
        return sim_vec

    def predict_event_types(self, ment_emb, mem_context, metadata_embs, span_score, gt_event_subtypes, follow_gt):
        if self.use_mem_context:
            input_vec = torch.cat([ment_emb, mem_context, metadata_embs, span_score], dim=0)
        else:
            input_vec = torch.cat([ment_emb, metadata_embs, span_score], dim=0)

        hidden_state = torch.unsqueeze(self.proj_layer(self.drop_module(input_vec)), dim=0)
        input_state = torch.zeros(1, self.emb_size).cuda()

        if follow_gt and len(gt_event_subtypes) < 3:
            gt_event_subtypes = gt_event_subtypes + [self.eos_event_id]

        logit_list = []
        all_pred_event_subtype_list = []
        # print(gt_event_subtypes)
        for i in range(3):
            hidden_state = self.event_subtype_rnn(input_state, hidden_state)
            event_subtype_logit = self.event_subtype_proj(torch.squeeze(hidden_state, dim=0))
            pred_event_subtype = torch.argmax(event_subtype_logit).item()

            logit_list.append(event_subtype_logit)

            all_pred_event_subtype_list.append(pred_event_subtype)

            if follow_gt:
                gt_event_subtype = gt_event_subtypes[i]
                if gt_event_subtype != self.eos_event_id:
                    input_state = self.event_subtype_embeddings(torch.tensor([gt_event_subtype]).long().cuda())
                else:
                    break
            else:
                if pred_event_subtype != self.eos_event_id:
                    input_state = self.event_subtype_embeddings(torch.tensor([pred_event_subtype]).long().cuda())
                else:
                    break

        return logit_list, all_pred_event_subtype_list, gt_event_subtypes

    def predict_action(self, ment_boundary, query_vector, local_emb, ment_type, ment_score, feature_embs):
        coref_new_scores = self.get_coref_new_scores(
            ment_boundary, query_vector, local_emb, ment_type, ment_score, feature_embs)

        # Negate the mention score
        not_a_ment_score = -ment_score
        # print(not_a_ment_score)
        over_ign_score = torch.cat([torch.tensor([0.0]).cuda(), not_a_ment_score], dim=0).cuda()
        return coref_new_scores, over_ign_score

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

    def forward(self, example, mention_emb_list, local_emb_list, mention_scores, pred_mentions,
                mention_to_cluster, rand_fl_list, teacher_forcing=False, random_threshold=1.0):
        # Initialize memory
        self.initialize_memory()

        sentence_map = example["sentence_map"]
        metadata = {'genre': example['doc_type']}

        type_logit_list = []
        type_list = []
        action_logit_list = []
        action_list = []  # argmax actions
        gt_actions = []
        first_overwrite = True
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        cluster_to_cell = {}
        cell_counter = 0

        for ment_idx, (raw_ment_emb, local_emb, (span_start, span_end), span_score) in enumerate(
                zip(mention_emb_list, local_emb_list, pred_mentions, mention_scores)):

            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            sent_idx = sentence_map[span_start]
            feature_embs = self.get_feature_embs(ment_idx, sent_idx, metadata)
            metadata_embs = self.get_metadata_embs(metadata)
            ment_boundary = (span_start, span_end)

            invalid_span = ment_boundary not in mention_to_cluster

            if not (follow_gt and invalid_span and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability
                mem_context = None
                if self.use_mem_context:
                    mem_context = self.get_mem_context(raw_ment_emb, feature_embs)

                gt_event_subtypes = {}
                if not invalid_span:
                    for (ment_type, ment_cluster) in mention_to_cluster[ment_boundary]:
                        gt_event_subtypes[ment_type] = ment_cluster

                gt_event_subtypes_list = sorted(list(gt_event_subtypes.keys()))

                import random
                use_gt_list = (follow_gt and random.random() < random_threshold)

                logit_list, all_pred_event_subtype_list, gt_event_subtypes_list = self.predict_event_types(
                    raw_ment_emb, mem_context, metadata_embs, span_score,
                    gt_event_subtypes_list, use_gt_list)

                all_pred_event_subtype_list = sorted(list(set(all_pred_event_subtype_list)))
                if use_gt_list:
                    event_subtype_list = gt_event_subtypes_list
                    # print(all_pred_event_subtype_list)
                    type_logit_list.extend(logit_list)
                    type_list.extend(gt_event_subtypes_list)
                else:
                    event_subtype_list = all_pred_event_subtype_list
                    # print("Using predicted subtypes")

                if self.eos_event_id in event_subtype_list:
                    # Check if EOS symbol i.e.
                    event_subtype_list.remove(self.eos_event_id)

                for subtype_idx, event_subtype in enumerate(event_subtype_list):
                    ment_score = torch.unsqueeze(logit_list[subtype_idx][event_subtype], dim=0)
                    event_type = get_event_type(event_subtype)
                    feature_embs = self.get_feature_embs(ment_idx, sent_idx, metadata)
                    ment_emb = torch.cat([
                        raw_ment_emb, self.event_subtype_embeddings(torch.tensor(event_subtype).long().cuda())])
                    coref_new_scores, overwrite_ign_scores = self.predict_action(
                        ment_boundary, ment_emb, local_emb, event_subtype, ment_score, feature_embs)

                    pred_cell_idx, pred_action_str = self.interpret_scores(
                        coref_new_scores, overwrite_ign_scores, first_overwrite)

                    if event_subtype in gt_event_subtypes:
                        ment_cluster = gt_event_subtypes[event_subtype]
                        if ment_cluster in cluster_to_cell:
                            gt_cell_idx = cluster_to_cell[ment_cluster]
                            gt_action_str = 'c'
                        else:
                            cluster_to_cell[ment_cluster] = cell_counter
                            gt_cell_idx = cell_counter
                            gt_action_str = 'o'
                            cell_counter += 1
                    else:
                        gt_cell_idx = -1
                        gt_action_str = 'i'

                    if follow_gt:
                        # Training - Operate over the ground truth
                        action_str = gt_action_str
                        cell_idx = gt_cell_idx
                    else:
                        # Inference time
                        action_str = pred_action_str
                        cell_idx = pred_cell_idx

                    # print(gt_cell_idx, gt_action_str, event_subtype, gt_event_subtypes)
                    action_logit_list.append((
                        coref_new_scores, overwrite_ign_scores, gt_cell_idx, gt_action_str))

                    gt_actions.append([gt_cell_idx, gt_action_str, (span_start, span_end, event_subtype)])

                    # print(pred_cell_idx, pred_action_str, gt_action_str, gt_cell_idx)
                    last_action_str = action_str
                    action_list.append((pred_cell_idx, pred_action_str, (span_start, span_end, event_subtype)))

                    if first_overwrite and action_str == 'o':
                        first_overwrite = False
                        # We start with a single empty memory cell
                        self.mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                        self.ent_counter = torch.tensor([1.0]).cuda()
                        self.last_mention_idx[0] = ment_idx
                        self.last_sent_idx[0] = sent_idx
                        self.cluster_type[0] = event_type
                        self.last_mention_boundary.append(ment_boundary)
                    else:
                        num_ents = self.mem_vectors.shape[0]
                        # Update the memory
                        cell_mask = (torch.arange(0, num_ents) == cell_idx).float().cuda()
                        mask = torch.unsqueeze(cell_mask, dim=1)
                        mask = mask.repeat(1, self.mem_size)

                        if action_str == 'c':
                            self.coref_update(ment_emb, local_emb, cell_idx, mask)
                            self.ent_counter = self.ent_counter + cell_mask
                            self.last_mention_idx[cell_idx] = ment_idx
                            self.last_sent_idx[cell_idx] = sent_idx
                            self.last_mention_boundary[cell_idx] = ment_boundary

                            if self.use_ment_type:
                                assert (event_type == self.cluster_type[cell_idx])
                        elif action_str == 'o':
                            # Append the new vector
                            self.mem_vectors = torch.cat([self.mem_vectors, torch.unsqueeze(ment_emb, dim=0)], dim=0)
                            self.ent_counter = torch.cat([self.ent_counter, torch.tensor([1.0]).cuda()], dim=0)
                            self.last_mention_idx = torch.cat([self.last_mention_idx, torch.tensor([ment_idx]).cuda()], dim=0)
                            self.last_sent_idx = torch.cat([self.last_sent_idx, torch.tensor([sent_idx]).cuda()], dim=0)
                            self.cluster_type = torch.cat([self.cluster_type, torch.tensor([event_type]).cuda()], dim=0)
                            self.last_mention_boundary.append(ment_boundary)

        from collections import Counter
        print(Counter([action[1] for action in gt_actions]))
        return type_logit_list, type_list, action_logit_list, action_list, gt_actions
