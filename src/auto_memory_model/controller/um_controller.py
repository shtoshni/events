import torch
import torch.nn as nn

from auto_memory_model.memory.um_memory import UnboundedMemory
from auto_memory_model.controller.base_controller import BaseController
from auto_memory_model.utils import get_mention_to_cluster
from red_utils.utils import get_doc_type
from pytorch_utils.label_smoothing import LabelSmoothingLoss


class UnboundedMemController(BaseController):
    def __init__(self, new_ent_wt=1.0, **kwargs):
        super(UnboundedMemController, self).__init__(**kwargs)
        self.new_ent_wt = new_ent_wt
        self.memory_net = UnboundedMemory(
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
            drop_module=self.drop_module, **kwargs)
        # Set loss functions
        self.loss_fn = {}
        if self.training:
            self.label_smoothing_fn = LabelSmoothingLoss(smoothing=self.label_smoothing_wt, dim=0)
        else:
            self.label_smoothing_fn = LabelSmoothingLoss(smoothing=0.0, dim=0)

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

            if action_str == 'c':
                gt_idx = cell_idx
            else:
                # Overwrite
                gt_idx = num_cells
                num_cells += 1


            # print(target, logit_tens.shape)
            weight = torch.ones_like(action_prob_list[idx]).float().cuda()
            weight[-1] = self.new_ent_wt

            # logit_tens = torch.unsqueeze(action_prob_list[idx], dim=0)
            # target = torch.tensor([gt_idx]).cuda()
            # coref_loss += torch.nn.functional.cross_entropy(input=logit_tens, target=target, weight=weight)

            # logit_tens = action_prob_list[idx]
            # smoothing_term = 0.1
            # target = torch.ones_like(logit_tens).cuda() * (smoothing_term/logit_tens.shape[0])
            # target[gt_idx] = 1 - smoothing_term
            #
            # coref_loss += torch.mean(torch.sum(-target * logit_tens))
            coref_loss += self.label_smoothing_fn(action_prob_list[idx], torch.tensor([gt_idx]).cuda())

        return coref_loss

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        gt_mentions, pred_mentions, gt_actions, mention_emb_list =\
            self.get_mention_embs_and_actions(example)

        if len(pred_mentions) > 0:
            doc_type = get_doc_type(example)
            action_prob_list, action_list = self.memory_net(
                doc_type, mention_emb_list, gt_actions, pred_mentions,
                teacher_forcing=teacher_forcing)  # , example[""])

            coref_loss = 0.0
            if self.training or teacher_forcing:
                loss = {}
                coref_loss = self.calculate_coref_loss(action_prob_list, gt_actions)
                loss['coref'] = coref_loss/len(mention_emb_list)
                loss['total'] = loss['coref']
                return loss, action_list, pred_mentions, gt_actions, gt_mentions
            else:
                return coref_loss, action_list, pred_mentions, gt_actions, gt_mentions

        return None
