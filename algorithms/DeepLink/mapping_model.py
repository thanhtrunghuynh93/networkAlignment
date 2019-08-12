import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F



def autoencoder_loss(decoded, source_feats, inversed_decoded, target_feats):
    num_examples1 = source_feats.shape[0]
    num_examples2 = target_feats.shape[0]
    straight_loss = (num_examples1 - (decoded * source_feats).sum())/num_examples1
    inversed_loss = (num_examples2 - (inversed_decoded * target_feats).sum())/num_examples2
    loss = straight_loss + inversed_loss
    return loss


class MappingModel(nn.Module):
    def __init__(self, embedding_dim=800, hidden_dim1=1200, hidden_dim2=1600, source_embedding=None, target_embedding=None):
        """
        Parameters
        ----------
        embedding_dim: int
            Embedding dim of nodes
        hidden_dim1: int
            Number of hidden neurons in the first hidden layer of MLP
        hidden_dim2: int
            Number of hidden neurons in the second hidden layer of MLP
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for source nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target nodes
        """

        super(MappingModel, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

        # theta is a MLP nn (encoder)
        self.theta = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])
        # inversed_theta is a MLP nn (decoder)
        self.inversed_theta = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])


    def forward(self, source_feats, mode='t'):
        encoded = self.theta(source_feats)
        encoded = F.normalize(encoded, dim=1)
        if mode != 't':
            return encoded
        decoded = self.inversed_theta(encoded)
        decoded = F.normalize(decoded, dim=1)
        return decoded


    def inversed_forward(self, target_feats):
        inversed_encoded = self.inversed_theta(target_feats)
        inversed_encoded = F.normalize(inversed_encoded, dim=1)
        inversed_decoded = self.theta(inversed_encoded)
        inversed_decoded = F.normalize(inversed_decoded, dim=1)
        return inversed_decoded


    def supervised_loss(self, source_batch, target_batch, alpha=1, k=5):
        source_feats = self.source_embedding[source_batch]
        target_feats = self.target_embedding[target_batch]

        source_after_map = self.theta(source_feats)
        source_after_map = F.normalize(source_after_map)

        target_after_map = self.inversed_theta(target_feats)
        target_after_map = F.normalize(target_after_map, dim=1)

        reward_source_target = 0
        reward_target_source = 0

        for i in range(source_feats.shape[0]):
            embedding_of_ua = source_feats[i]
            embedding_of_target_of_ua = target_feats[i]
            embedding_of_ua_after_map = source_after_map[i]
            reward_source_target += torch.sum(embedding_of_ua_after_map * embedding_of_target_of_ua)
            top_k_simi = self.find_topk_simi(embedding_of_target_of_ua, self.target_embedding, k=k)
            reward_source_target += self.compute_rst(embedding_of_ua_after_map, top_k_simi)
            reward_target_source += self.compute_rts(embedding_of_ua, top_k_simi)
        st_loss = -alpha*reward_source_target/source_feats.shape[0]
        ts_loss = -(1-alpha)*reward_target_source/target_feats.shape[0]
        loss = st_loss + ts_loss
        return loss


    def unsupervised_loss(self, source_batch, target_batch):
        source_feats = self.source_embedding[source_batch]
        target_feats = self.target_embedding[target_batch]
        decoded = self.forward(source_feats)
        inversed_decoded = self.inversed_forward(target_feats)
        loss = autoencoder_loss(decoded, source_feats, inversed_decoded, target_feats)
        return loss


    def compute_rst(self, embedding_of_ua_after_map, top_k_simi):
        top_k_embedding = self.target_embedding[top_k_simi]
        cosin = torch.sum(embedding_of_ua_after_map * top_k_embedding, dim=1)
        reward = torch.mean(torch.log(cosin + 1))
        return reward


    def compute_rts(self, embedding_of_ua, top_k_simi):
        top_k_embedding = self.target_embedding[top_k_simi]
        top_k_simi_after_inversed_map = self.inversed_theta(top_k_embedding)
        top_k_simi_after_inversed_map = F.normalize(top_k_simi_after_inversed_map, dim=1)
        cosin = torch.sum(embedding_of_ua * top_k_simi_after_inversed_map, dim=1)
        reward = torch.mean(torch.log(cosin + 1))
        return reward


    def find_topk_simi(self, embedding_of_ua_after_map, target_embedding, k):
        cosin_simi_matrix = torch.matmul(embedding_of_ua_after_map, target_embedding.t())
        top_k_index = cosin_simi_matrix.sort()[1][-k:]
        return top_k_index



