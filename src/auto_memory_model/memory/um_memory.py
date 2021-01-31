import torch
import torch.nn as nn
from auto_memory_model.memory.base_memory import BaseMemory
from kbp_2015_utils.constants import EVENT_SUBTYPES
from kbp_2015_utils.utils import get_event_type
from pytorch_utils.modules import MLP



class UnboundedMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemory, self).__init__(**kwargs)

        self.mem_sim_mlp = MLP(self.mem_size + self.hsize + self.num_feats * self.emb_size, self.mlp_size, 1,
                               num_hidden_layers=self.mlp_depth, bias=True, drop_module=self.drop_module)

        self.num_type_mlp = MLP(self.mem_size + self.hsize + 2 * self.emb_size + 1, self.mlp_size, output_size=1,
                                num_hidden_layers=self.mlp_depth, bias=True, drop_module=self.drop_module)

        self.event_subtype_mlp = MLP(self.mem_size + self.hsize + 2 * self.emb_size, self.mlp_size,
                                     output_size=len(EVENT_SUBTYPES), num_hidden_layers=self.mlp_depth, bias=True,
                                     drop_module=self.drop_module)

        self.sim_softmax = nn.Softmax(dim=0)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        self.mem_vectors = torch.zeros(1, self.mem_size).cuda()
        self.local_vectors = torch.zeros(1, self.mem_size).cuda()
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

    def predict_num_types(self, ment_emb, mem_context, metadata_embs, ment_score):
        input_vec = torch.cat([ment_emb, mem_context, metadata_embs, ment_score], dim=0)
        num_type_logit = self.num_type_mlp(input_vec)
        num_types_cont = 3 * torch.sigmoid(num_type_logit)
        # num_type_logit[0] -= ment_score

        return num_types_cont, int(torch.round(num_types_cont).item())

    def predict_types(self, ment_emb, mem_context, metadata_embs):
        input_vec = torch.cat([ment_emb, mem_context, metadata_embs], dim=0)
        event_subtype_logits = self.event_subtype_mlp(input_vec)

        return event_subtype_logits

    def interpret_scores(self, coref_new_scores, first_overwrite):
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
            return num_ents, 'o'
        else:
            raise NotImplementedError

    def forward(self, example, mention_emb_list, local_emb_list, mention_scores, pred_mentions,
                gt_actions, rand_fl_list, teacher_forcing=False):
        # Initialize memory
        self.initialize_memory()

        sentence_map = example["sentence_map"]
        metadata = {'genre': example['doc_type']}

        num_logit_list = []
        type_logit_list = []
        action_logit_list = []
        action_list = []  # argmax actions
        first_overwrite = True
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        for ment_idx, (raw_ment_emb, local_emb, (span_start, span_end), ment_score, gt_action_list) in enumerate(
                zip(mention_emb_list, local_emb_list, pred_mentions, mention_scores, gt_actions)):

            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            sent_idx = sentence_map[span_start]
            feature_embs = self.get_feature_embs(ment_idx, sent_idx, metadata)
            metadata_embs = self.get_metadata_embs(metadata)
            ment_boundary = (span_start, span_end)

            if not (follow_gt and gt_action_list is None and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability
                mem_context = self.get_mem_context(raw_ment_emb, feature_embs)
                num_type_logit, pred_num_types = self.predict_num_types(
                    raw_ment_emb, mem_context, metadata_embs, ment_score)

                gt_num_types = (0 if gt_action_list is None else len(gt_action_list))
                # print(gt_num_types - pred_num_types)
                event_subtype_logits = self.predict_types(raw_ment_emb, mem_context, metadata_embs)

                gt_ment_type_list = ([] if gt_action_list is None
                                     else [gt_action[2][2] for gt_action in gt_action_list])
                # print(gt_ment_type_list, gt_action_list)
                if follow_gt:
                    # Training - Operate over the ground truth
                    event_subtypes = gt_ment_type_list
                else:
                    # Inference time
                    if pred_num_types > 0:
                        event_subtypes = sorted(torch.topk(event_subtype_logits, k=pred_num_types)[1].tolist())
                    else:
                        event_subtypes = []

                num_logit_list.append((gt_num_types, num_type_logit))
                type_logit_list.append((gt_ment_type_list, event_subtype_logits))

                # print(event_subtypes, self.mem_vectors.shape[0])
                for subtype_idx, event_subtype in enumerate(event_subtypes):

                    # print(event_subtype, gt_cell_idx, gt_action_str)

                    event_type = get_event_type(event_subtype)
                    feature_embs = self.get_feature_embs(ment_idx, sent_idx, metadata)
                    ment_emb = torch.cat([
                        raw_ment_emb, self.event_subtype_embeddings(torch.tensor(event_subtype).long().cuda())])
                    coref_new_scores = self.get_coref_new_scores(
                        ment_boundary, ment_emb, local_emb, event_subtype, feature_embs)

                    pred_cell_idx, pred_action_str = self.interpret_scores(
                        coref_new_scores, first_overwrite)
                    if follow_gt:
                        # Training - Operate over the ground truth
                        gt_cell_idx, gt_action_str, _ = gt_action_list[subtype_idx]
                        action_logit_list.append((coref_new_scores, gt_cell_idx, gt_action_str))
                        action_str = gt_action_str
                        cell_idx = gt_cell_idx

                    action_list.append((pred_cell_idx, pred_action_str, (span_start, span_end, event_subtype)))

                    if not follow_gt:
                        # Inference time
                        action_str = pred_action_str
                        cell_idx = pred_cell_idx

                    last_action_str = action_str

                    if first_overwrite and action_str == 'o':
                        first_overwrite = False
                        # We start with a single empty memory cell
                        self.mem_vectors = torch.unsqueeze(ment_emb, dim=0)
                        if self.use_srl or self.use_local_attention:
                            self.local_vectors = torch.unsqueeze(local_emb, dim=0)
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

        return num_logit_list, type_logit_list, action_logit_list, action_list
