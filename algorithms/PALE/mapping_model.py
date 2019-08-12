from algorithms.PALE.loss import MappingLossFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################
#               MAPPING MODELS
##############################################################################

class PaleMapping(nn.Module):
    def __init__(self, source_embedding, target_embedding):
        """
        Parameters
        ----------
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target_nodes
        target_neighbor: dict
            dict of target_node -> target_nodes_neighbors. Used for calculate vinh_loss
        """

        super(PaleMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.loss_fn = MappingLossFunctions()
    

class PaleMappingLinear(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding):
        super(PaleMappingLinear, self).__init__(source_embedding, target_embedding)
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        
        return mapping_loss

    def forward(self, source_feats):
        ret = self.maps(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret


class PaleMappingMlp(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding, activate_function='sigmoid'):

        super(PaleMappingMlp, self).__init__(source_embedding, target_embedding)

        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()

        hidden_dim = 2*embedding_dim
        self.mlp = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            self.activate_function,
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])


    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        

        return mapping_loss

    def forward(self, source_feats):
        ret = self.mlp(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret


