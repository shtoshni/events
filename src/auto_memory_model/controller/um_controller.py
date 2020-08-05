import torch
import torch.nn as nn

from auto_memory_model.memory.um_memory import UnboundedMemory
from auto_memory_model.controller.base_controller import BaseController
from auto_memory_model.utils import get_mention_to_cluster, get_ordered_mentions


class UnboundedMemController(BaseController):
    def __init__(self, new_ent_wt=1.0, **kwargs):
        super(UnboundedMemController, self).__init__(**kwargs)
        self.new_ent_wt = new_ent_wt
        self.memory_net = UnboundedMemory(
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
            drop_module=self.drop_module, **kwargs)
        # Set loss functions
        self.loss_fn = {}

    @staticmethod
    def get_actions(mentions, clusters):
        # Useful data structures
        mention_to_cluster = get_mention_to_cluster(clusters)

        actions = []
        cell_to_cluster = {}
        cluster_to_cell = {}

        cell_counter = 0
        for mention in mentions:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster in cluster_to_cell:
                # Cluster is already being tracked
                actions.append((cluster_to_cell[mention_cluster], 'c'))
            else:
                # Cluster is not being tracked
                # Add the mention to being tracked
                cluster_to_cell[mention_cluster] = cell_counter
                cell_to_cluster[cell_counter] = mention_cluster
                actions.append((cell_counter, 'o'))
                cell_counter += 1

        return actions

    def calculate_coref_loss(self, action_prob_list, action_tuple_list):
        num_cells = 1
        coref_loss = 0.0

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if idx == 0:
                continue

            gt_idx = None
            if action_str == 'c':
                gt_idx = cell_idx
            else:
                # Overwrite
                gt_idx = num_cells
                num_cells += 1

            target = torch.tensor([gt_idx]).cuda()
            logit_tens = torch.unsqueeze(action_prob_list[idx], dim=0)

            # print(target, logit_tens.shape)
            weight = torch.ones_like(action_prob_list[idx]).float().cuda()
            weight[-1] = self.new_ent_wt
            coref_loss += torch.nn.functional.cross_entropy(input=logit_tens, target=target, weight=weight)

        return coref_loss

    def calculate_span_type_loss(self, span_type_logit_list, span_type_list):
        span_type_logit_tensor = torch.stack(span_type_logit_list)
        span_type_tensor = torch.tensor(span_type_list).cuda()

        span_type_pred = torch.argmax(span_type_logit_tensor, dim=1)
        span_type_corr = torch.sum(span_type_pred == span_type_tensor).item()

        span_type_loss = torch.nn.functional.cross_entropy(
            input=span_type_logit_tensor, target=span_type_tensor)
        return span_type_loss, span_type_corr

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        gt_mentions, pred_mentions, gt_actions, mention_emb_list =\
            self.get_mention_embs_and_actions(example)

        doc_type = example["doc_key"].split("/")[0]
        action_prob_list, action_list, span_type_logit_list, span_type_list = self.memory_net(
            doc_type, mention_emb_list, gt_actions, pred_mentions,
            teacher_forcing=teacher_forcing)  # , example[""])

        span_type_loss, span_type_corr = self.calculate_span_type_loss(
            span_type_logit_list, span_type_list)
        span_type_total = len(span_type_list)

        loss = {}
        loss['span_type'] = span_type_loss
        if self.training or teacher_forcing:
            coref_loss = self.calculate_coref_loss(action_prob_list, gt_actions)
            loss['coref'] = coref_loss/len(mention_emb_list)
            return (loss, action_list, pred_mentions, gt_actions, gt_mentions,
                    span_type_corr, span_type_total)
        else:
            return (loss, action_list, pred_mentions, gt_actions, gt_mentions,
                    span_type_corr, span_type_total)
