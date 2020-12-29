import torch
from auto_memory_model.memory.base_fixed_memory import BaseMemory
from kbp_2015_utils.utils import get_event_type


class UnboundedMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemory, self).__init__(**kwargs)
        self.mem_vectors = torch.zeros(1, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0.0]).cuda()
        self.last_mention_idx = torch.zeros(1).long().cuda()
        self.last_sent_idx = torch.zeros(1).long().cuda()
        self.cluster_type = torch.tensor([-1]).cuda()
        self.cluster_subtype_emb = torch.zeros(1, self.emb_size).cuda()

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        self.mem_vectors = torch.zeros(1, self.mem_size).cuda()
        self.ent_counter = torch.tensor([0.0]).cuda()
        self.last_mention_idx = torch.zeros(1).long().cuda()
        self.last_sent_idx = torch.zeros(1).long().cuda()
        self.cluster_type = torch.tensor([-1]).cuda()
        self.cluster_subtype_emb = torch.zeros(1, self.emb_size).cuda()

    def predict_action(self, query_vector, ment_type, ment_score, feature_embs):
        coref_new_scores = self.get_coref_new_scores(query_vector, ment_type, ment_score, feature_embs)

        # Negate the mention score
        not_a_ment_score = -ment_score

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

    def forward(self, example, mention_emb_list, mention_scores, pred_mentions, gt_actions, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        self.initialize_memory()

        sentence_map = example["sentence_map"]
        metadata = {'genre': example['doc_type']}

        action_logit_list = []
        action_list = []  # argmax actions
        first_overwrite = True
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        for ment_idx, (ment_emb,  (span_start, span_end, event_subtype), ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, pred_mentions, mention_scores, gt_actions)):
            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            feature_embs = self.get_feature_embs(ment_idx, metadata)
            event_type = get_event_type(event_subtype)

            sent_idx = sentence_map[span_start]

            # if self.use_ment_type:
            #     query_vector = ment_emb
            # else:
            #     query_vector = self.query_projector(torch.cat([ment_emb, ment_type_emb], dim=0))

            # query_vector = self.query_projector(torch.cat([ment_emb, ment_type_emb], dim=0))
            query_vector = ment_emb

            if not (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability
                coref_new_scores, overwrite_ign_scores = self.predict_action(
                    query_vector, event_subtype, ment_score, feature_embs)

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
            event_subtype_emb = self.event_subtype_embeddings(torch.tensor(event_subtype).long().cuda())

            if first_overwrite and action_str == 'o':
                first_overwrite = False
                # We start with a single empty memory cell
                self.mem_vectors = torch.unsqueeze(query_vector, dim=0)
                self.ent_counter = torch.tensor([1.0]).cuda()
                self.last_mention_idx[0] = ment_idx
                self.last_sent_idx[0] = sent_idx
                self.cluster_type[0] = event_type
                self.cluster_subtype_emb[0] = event_subtype_emb
            else:
                num_ents = self.mem_vectors.shape[0]
                # Update the memory
                cell_mask = (torch.arange(0, num_ents) == cell_idx).float().cuda()
                mask = torch.unsqueeze(cell_mask, dim=1)
                mask = mask.repeat(1, self.mem_size)
                subtype_mask = torch.unsqueeze(cell_mask, dim=1).repeat(1, self.emb_size)

                # print(cell_idx, action_str, mem_vectors.shape[0])
                if action_str == 'c':
                    self.coref_update(query_vector, event_subtype_emb, cell_idx, mask, subtype_mask)
                    self.ent_counter = self.ent_counter + cell_mask
                    self.last_mention_idx[cell_idx] = ment_idx
                    self.last_sent_idx[cell_idx] = sent_idx

                    if self.use_ment_type:
                        assert (event_type == self.cluster_type[cell_idx])
                elif action_str == 'o':
                    # Append the new vector
                    self.mem_vectors = torch.cat([self.mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)
                    self.ent_counter = torch.cat([self.ent_counter, torch.tensor([1.0]).cuda()], dim=0)
                    self.last_mention_idx = torch.cat([self.last_mention_idx, torch.tensor([ment_idx]).cuda()], dim=0)
                    self.last_sent_idx = torch.cat([self.last_sent_idx, torch.tensor([sent_idx]).cuda()], dim=0)
                    self.cluster_type = torch.cat([self.cluster_type, torch.tensor([event_type]).cuda()], dim=0)
                    self.cluster_subtype_emb = torch.cat([self.cluster_subtype_emb,
                                                          torch.unsqueeze(event_subtype_emb, dim=0)], dim=0)

        return action_logit_list, action_list
