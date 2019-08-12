from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.DeepLink.mapping_model import MappingModel
from algorithms.DeepLink.embedding_model import DeepWalk

from input.dataset import Dataset
from utils.graph_utils import load_gt

import numpy as np
import torch.nn as nn
import torch
import networkx as nx

import argparse
import time



class DeepLink(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        """
        Parameters
        ----------
        source_dataset: Dataset
            Dataset object of source dataset
        target_dataset: Dataset
            Dataset object of target dataset
        args: argparse.ArgumentParser.parse_args()
            arguments as parameters for model.
        """
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        super(DeepLink, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alpha = args.alpha
        self.map_batchsize = args.batch_size_mapping
        self.cuda = args.cuda
        self.embedding_dim = args.embedding_dim
        self.embedding_epochs = args.embedding_epochs
        self.supervised_epochs = args.supervised_epochs
        self.unsupervised_epochs = args.unsupervised_epochs
        self.supervised_lr = args.supervised_lr
        self.unsupervised_lr = args.unsupervised_lr
        self.num_cores = args.num_cores

        gt = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.full_gt = {}
        self.full_gt.update(gt)
        test_gt = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.full_gt.update(test_gt)
        self.full_gt = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in self.full_gt.items()}
        self.train_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k,v in gt.items()}


        self.number_walks = args.number_walks
        self.format = args.format
        self.walk_length = args.walk_length
        self.window_size = args.window_size
        self.top_k = args.top_k


        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_after_mapping = None
        self.source_train_nodes = np.array(list(self.train_dict.keys()))
        self.source_anchor_nodes = np.array(list(self.train_dict.keys()))

        self.hidden_dim1 = args.hidden_dim1
        self.hidden_dim2 = args.hidden_dim2
        self.seed = args.seed



    def get_alignment_matrix(self):
        return self.S

    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding

    def align(self):
        self.learn_embeddings()

        mapping_model = MappingModel(
                                embedding_dim=self.embedding_dim,
                                hidden_dim1=self.hidden_dim1,
                                hidden_dim2=self.hidden_dim2,
                                source_embedding=self.source_embedding,
                                target_embedding=self.target_embedding
                                )

        if self.cuda:
            mapping_model = mapping_model.cuda()

        m_optimizer_us = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr = self.unsupervised_lr)

        self.mapping_train_(mapping_model, m_optimizer_us, 'us')

        m_optimizer_s = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr = self.supervised_lr)
        self.mapping_train_(mapping_model, m_optimizer_s, 's')
        self.source_after_mapping = mapping_model(self.source_embedding, 'val')
        self.S = torch.matmul(self.source_after_mapping, self.target_embedding.t())

        self.S = self.S.detach().cpu().numpy()
        return self.S


    def mapping_train_(self, model, optimizer, mode='s'):
        if mode == 's':
            source_train_nodes = self.source_train_nodes
        else:
            source_train_nodes = self.source_anchor_nodes

        batch_size = self.map_batchsize
        n_iters = len(source_train_nodes)//batch_size
        assert n_iters > 0, "batch_size is too large"
        if(len(source_train_nodes) % batch_size > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        train_dict = None
        if mode == 's':
            n_epochs = self.supervised_epochs
            train_dict = self.train_dict
        else:
            n_epochs = self.unsupervised_epochs
            train_dict = self.full_gt


        for epoch in range(1, n_epochs+1):
            # for evaluate time
            start = time.time()

            print("Epoch {0}".format(epoch))
            np.random.shuffle(source_train_nodes)
            for iter in range(n_iters):
                source_batch = source_train_nodes[iter*batch_size:(iter+1)*batch_size]
                target_batch = [train_dict[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                start_time = time.time()
                if mode == 'us':
                    loss = model.unsupervised_loss(source_batch, target_batch)
                else:
                    loss = model.supervised_loss(source_batch, target_batch, alpha=self.alpha, k=self.top_k)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.5f}".format(loss.item()),
                          )
            
                total_steps += 1
            if mode == "s":
                self.s_mapping_epoch_time = time.time() - start
            else:
                self.un_mapping_epoch_time = time.time() - start

    def learn_embeddings(self):
        print("Start embedding for source nodes, using deepwalk")

        # for evaluate time
        start = time.time()

        source_embedding_model = DeepWalk(self.source_dataset.G, self.source_dataset.id2idx, self.number_walks, \
                        self.walk_length, self.window_size, self.embedding_dim, self.num_cores, self.embedding_epochs, seed=self.seed)

        self.source_embedding = torch.Tensor(source_embedding_model.get_embedding())

        self.embedding_epoch_time = time.time() - start

        print("Start embedding for target nodes, using deepwalk")

        target_embedding_model = DeepWalk(self.target_dataset.G, self.target_dataset.id2idx, self.number_walks, \
                        self.walk_length, self.window_size, self.embedding_dim, self.num_cores, self.embedding_epochs, seed=self.seed)

        self.target_embedding = torch.Tensor(target_embedding_model.get_embedding())
        if self.cuda:
            self.source_embedding = self.source_embedding.cuda()
            self.target_embedding = self.target_embedding.cuda()

