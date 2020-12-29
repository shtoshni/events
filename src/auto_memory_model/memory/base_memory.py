import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
import math
from kbp_2015_utils.constants import EVENT_SUBTYPES, EVENT_TYPES
from kbp_2015_utils.utils import get_event_type


LOG2 = math.log(2)


class BaseMemory(nn.Module):
    def __init__(self, hsize=300, mlp_size=200, mlp_depth=1, drop_module=None,
                 emb_size=20, entity_rep='max', dataset='kbp_2015', sample_invalid=1.0,
                 use_ment_type=False, use_doc_type=False,
                 **kwargs):
        super(BaseMemory, self).__init__()
        self.dataset = dataset

        self.sample_invalid = sample_invalid
        self.use_ment_type = use_ment_type
        self.use_doc_type = use_doc_type

        if self.use_doc_type:
            self.num_feats = 5
        else:
            self.num_feats = 4

        self.hsize = hsize
        self.mem_size = hsize
        self.mlp_size = mlp_size
        self.mlp_depth = mlp_depth
        self.emb_size = emb_size
        self.entity_rep = entity_rep

        self.drop_module = drop_module

        # 4 Actions + 1 Dummy start action
        # c = coref, o = overwrite, i = invalid, n = no space (ignore)
        self.action_str_to_idx = {'c': 0, 'o': 1, 'i': 2, 'n': 3, '<s>': 4}
        self.action_idx_to_str = ['c', 'o', 'i', 'n', '<s>']

        # if not self.use_ment_type:
        # self.query_projector = MLP(self.hsize + self.emb_size, self.mlp_size, self.mem_size,
        #                            num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)

        self.mem_coref_mlp = MLP(3 * self.mem_size + self.num_feats * self.emb_size, self.mlp_size, 1,
                                 num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)
        self.event_subtype_match_mlp = MLP(2 * self.emb_size, self.mlp_size, 1,
                                           num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)

        if self.entity_rep == 'learned_avg':
            self.alpha = MLP(2 * self.mem_size, 300, 1, num_hidden_layers=1, bias=True, drop_module=drop_module)

        self.last_action_embeddings = nn.Embedding(5, self.emb_size)

        self.doc_type_emb = nn.Embedding(2, self.emb_size)
        self.ment_dist_embeddings = nn.Embedding(10, self.emb_size)
        self.sent_dist_embeddings = nn.Embedding(10, self.emb_size)
        self.counter_embeddings = nn.Embedding(10, self.emb_size)
        self.event_subtype_embeddings = nn.Embedding(len(EVENT_SUBTYPES), self.emb_size)

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

    def get_ment_distance_emb(self, distance):
        distance_tens = self.get_distance_bucket(distance)
        distance_embs = self.ment_dist_embeddings(distance_tens)
        return distance_embs

    def get_sent_distance_emb(self, distance):
        distance_tens = self.get_distance_bucket(distance)
        distance_embs = self.sent_dist_embeddings(distance_tens)
        return distance_embs

    def get_counter_emb(self, ent_counter):
        counter_buckets = self.get_counter_bucket(ent_counter.long())
        counter_embs = self.counter_embeddings(counter_buckets)
        return counter_embs

    def get_last_action_emb(self, action_str):
        action_emb = self.action_str_to_idx[action_str]
        return self.last_action_emb(torch.tensor(action_emb).cuda())

    def get_coref_mask(self, ment_type):
        cell_mask = (self.ent_counter > 0.0).float()
        if self.use_ment_type:
            type_mask = (torch.tensor(ment_type).cuda() == self.cluster_type).float().cuda()
            return cell_mask * type_mask

        return cell_mask

    def get_feature_embs(self, ment_idx, metadata):
        ment_dist_embs = self.get_ment_distance_emb(ment_idx - self.last_mention_idx)
        sent_dist_embs = self.get_sent_distance_emb(ment_idx - self.last_sent_idx)

        counter_embs = self.get_counter_emb(self.ent_counter)

        feature_embs_list = [ment_dist_embs, sent_dist_embs, counter_embs]
        num_cells = ment_dist_embs.shape[0]

        if self.use_doc_type:
            if 'genre' in metadata:
                genre_emb = self.doc_type_emb(torch.tensor(metadata['genre']).long().cuda())
                genre_emb = torch.unsqueeze(genre_emb, dim=0).repeat(num_cells, 1)
                feature_embs_list.append(genre_emb)
            else:
                feature_embs_list.append(torch.zeros(1, self.emb_size).long().cuda())

        if 'last_action' in metadata:
            last_action_idx = torch.tensor(metadata['last_action']).long().cuda()
            last_action_emb = self.last_action_embeddings(last_action_idx)
            last_action_emb = torch.unsqueeze(last_action_emb, dim=0).repeat(num_cells, 1)
            feature_embs_list.append(last_action_emb)

        feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
        return feature_embs

    def get_ment_feature_embs(self, metadata):
        # Bucket is 0 for both the embeddings
        distance_embs = self.distance_embeddings(torch.tensor(0).cuda())
        counter_embs = self.counter_embeddings(torch.tensor(0).cuda())

        feature_embs_list = [distance_embs, counter_embs]

        if 'genre' in metadata:
            genre_emb = metadata['genre']
            feature_embs_list.append(genre_emb)

        if 'last_action' in metadata:
            last_action_idx = torch.tensor(metadata['last_action']).long().cuda()
            last_action_emb = self.last_action_embeddings(last_action_idx)
            feature_embs_list.append(last_action_emb)

        feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
        return feature_embs

    def get_coref_new_scores(self, query_vector, event_subtype, ment_score, feature_embs):
        # Repeat the query vector for comparison against all cells
        num_cells = self.mem_vectors.shape[0]
        rep_query_vector = query_vector.repeat(num_cells, 1)  # M x H

        # Event Subtype
        event_type = get_event_type(event_subtype)
        event_subtype_emb = self.event_subtype_embeddings(torch.tensor(event_subtype).long().cuda())
        rep_event_subtype = torch.unsqueeze(event_subtype_emb, dim=0).repeat(num_cells, 1)

        # Coref Score
        pair_vec = torch.cat([self.mem_vectors, rep_query_vector, self.mem_vectors * rep_query_vector,
                              feature_embs], dim=-1)
        pair_score = self.mem_coref_mlp(pair_vec)

        # Event Subtype Match Score
        # print(event_subtype)
        # hadamard_term = self.cluster_subtype_emb * rep_event_subtype
        subtype_pair_vec = torch.cat([self.cluster_subtype_emb, rep_event_subtype], dim=-1)  #, hadamard_term], dim=-1)
        subtype_pair_score = self.event_subtype_match_mlp(subtype_pair_vec)

        coref_score = torch.squeeze(pair_score + subtype_pair_score, dim=-1) + ment_score  # M

        # Event type used for coreference mask
        coref_new_mask = torch.cat([self.get_coref_mask(event_type), torch.tensor([1.0]).cuda()], dim=0)
        coref_new_scores = torch.cat(([coref_score, torch.tensor([0.0]).cuda()]), dim=0)

        coref_new_not_scores = coref_new_scores * coref_new_mask + (1 - coref_new_mask) * (-1e4)
        return coref_new_not_scores

    def coref_update(self, query_vector, event_subtype_emb, cell_idx, mask, subtype_mask):
        if self.entity_rep == 'learned_avg':
            alpha_wt = torch.sigmoid(
                self.alpha(torch.cat([self.mem_vectors[cell_idx, :], query_vector], dim=0)))
            avg_pool_vec = alpha_wt * self.mem_vectors[cell_idx, :] + (1 - alpha_wt) * query_vector
            self.mem_vectors = self.mem_vectors * (1 - mask) + mask * torch.unsqueeze(avg_pool_vec, dim=0)
        else:
            total_counts = torch.unsqueeze((self.ent_counter + 1).float(), dim=1)
            pool_vec_num = self.mem_vectors * torch.unsqueeze(self.ent_counter, dim=1) + query_vector
            avg_pool_vec = pool_vec_num / total_counts
            self.mem_vectors = self.mem_vectors * (1 - mask) + mask * avg_pool_vec

            pool_subtype_num = self.cluster_subtype_emb * torch.unsqueeze(self.ent_counter, dim=1) + event_subtype_emb
            avg_subtype_vec = pool_subtype_num / total_counts
            self.cluster_subtype_emb = self.cluster_subtype_emb * (1 - subtype_mask) + subtype_mask * avg_subtype_vec


