import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingLossFunctions(object):
    def __init__(self, loss_fn='xent', neg_sample_weights=1.0):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be
                based on dot product.
        """
        self.neg_sample_weights = neg_sample_weights
        self.output_dim = 1
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        else:
            print("Not implemented yet.")


    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [n_batch_edges x feature_size].
        """
        # shape: [n_batch_edges, input_dim1]
        result = torch.sum(inputs1 * inputs2, dim=1) # shape: (n_batch_edges,)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [n_batch_edges x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        neg_aff = inputs1.mm(neg_samples.t()) #(n_batch_edges, num_neg_samples)
        return neg_aff


    def sigmoid_cross_entropy_with_logits(self, labels, logits):
        sig_aff = torch.sigmoid(logits)
        loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)
        return loss

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        inputs1: Tensor (512, 256), normalized vector
        inputs2: Tensor (512, 256), normalized vector
        neg_sample: Tensor (20, 256)
        """
        cuda = inputs1.is_cuda
        true_aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples)
        true_labels = torch.ones(true_aff.shape)  # (n_batch_edges,)
        if cuda:
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if cuda:
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        loss0 = true_xent.sum()
        loss1 = self.neg_sample_weights * neg_xent.sum()
        loss = loss0 + loss1
        return loss, loss0, loss1


class MappingLossFunctions(object):
    def __init__(self):
        self.loss_fn = self._euclidean_loss

    def loss(self, inputs1, inputs2):
        return self.loss_fn(inputs1, inputs2)

    def _euclidean_loss(self, inputs1, inputs2):
        sub = inputs2 - inputs1
        square_sub = sub**2
        loss = torch.sum(square_sub)        
        return loss
