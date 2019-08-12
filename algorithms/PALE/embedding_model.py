from algorithms.PALE.loss import EmbeddingLossFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled


class PaleEmbedding(nn.Module):
    def __init__(self, n_nodes, embedding_dim, deg, neg_sample_size, cuda):

        """
        Parameters
        ----------
        n_nodes: int
            Number of all nodes
        embedding_dim: int
            Embedding dim of nodes
        deg: ndarray , shape = (-1,)
            Array of degrees of all nodes
        neg_sample_size : int
            Number of negative candidate to sample
        cuda: bool
            Whether to use cuda
        """

        super(PaleEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.deg = deg
        self.neg_sample_size = neg_sample_size
        self.link_pred_layer = EmbeddingLossFunctions()
        self.n_nodes = n_nodes
        self.use_cuda = cuda


    def loss(self, nodes, neighbor_nodes):
        batch_output, neighbor_output, neg_output = self.forward(nodes, neighbor_nodes)
        batch_size = batch_output.shape[0]
        loss, loss0, loss1 = self.link_pred_layer.loss(batch_output, neighbor_output, neg_output)
        loss = loss/batch_size
        loss0 = loss0/batch_size
        loss1 = loss1/batch_size
        
        return loss, loss0, loss1


    def forward(self, nodes, neighbor_nodes=None):
        node_output = self.node_embedding(nodes)
        node_output = F.normalize(node_output, dim=1)

        if neighbor_nodes is not None:
            neg = fixed_unigram_candidate_sampler(
                num_sampled=self.neg_sample_size,
                unique=False,
                range_max=len(self.deg),
                distortion=0.75,
                unigrams=self.deg
                )

            neg = torch.LongTensor(neg)
            
            if self.use_cuda:
                neg = neg.cuda()
            neighbor_output = self.node_embedding(neighbor_nodes)
            neg_output = self.node_embedding(neg)
            # normalize
            neighbor_output = F.normalize(neighbor_output, dim=1)
            neg_output = F.normalize(neg_output, dim=1)

            return node_output, neighbor_output, neg_output

        return node_output

    def get_embedding(self):
        nodes = np.arange(self.n_nodes)
        nodes = torch.LongTensor(nodes)
        if self.use_cuda:
            nodes = nodes.cuda()
        embedding = None
        BATCH_SIZE = 512
        for i in range(0, self.n_nodes, BATCH_SIZE):
            j = min(i + BATCH_SIZE, self.n_nodes)
            batch_nodes = nodes[i:j]
            if batch_nodes.shape[0] == 0: break
            batch_node_embeddings = self.forward(batch_nodes)
            if embedding is None:
                embedding = batch_node_embeddings
            else:
                embedding = torch.cat((embedding, batch_node_embeddings))

        return embedding
