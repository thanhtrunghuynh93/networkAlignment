from __future__ import print_function, division
import numpy as np
import random
import json
import sys
import os
import argparse
from shutil import copyfile

import networkx as nx
from networkx.readwrite import json_graph
import pdb


def create_permutation_matrix(shape):
    L = list(range(shape[1]))
    P = np.zeros(shape)
    for i in range(P.shape[0]):
        a = np.random.choice(L, 1)
        P[i][a] = 1
        L.remove(a)
    return P


def permutation(A, P):
    return np.dot(P, np.dot(A, P.T))


def get_degree(A):
    return np.sum(A, axis=1)


def add_noise_to_graphstructure(A, ps):
    degree = get_degree(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                dice = np.random.rand()
                if dice <= ps and degree[i] > 1 and degree[j] > 1:
                    A[i,j] = 0
                    A[j,i] = 0
                    degree[i] -= 1
                    degree[j] -= 1
    return A


def generate_node_feature(num_nodes, num_feats):
    features = np.zeros((num_nodes, num_feats), dtype=int)
    for i in range(num_nodes):
        a = np.random.randint(0, num_feats)
        features[i, a] = 1

    iden_feat = np.identity(num_feats, dtype=int)
    L = list(range(num_nodes))
    for i in range(len(iden_feat)):
        j = np.random.choice(L)
        features[j] = iden_feat[i]
        L.remove(j)

    return features

def add_noise_to_node_features(features, pa):
    features1 = deepcopy(features)
    list_feat = [features1[0]]
    for i in range(features1):
        notin = 1
        for feat in list_feat:
            if (feat==features1[i]).all():
                notin = 0
        if notin:
            list_feat.append(features1[i])
    for node in range(len(features1)):
        if(np.random.rand()< pa):
            j = np.random.randint(0, len(list_feat))
            features1[i] = list_feat[j]
    return features, features1

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--input', default=None, help='Path to load data')
    parser.add_argument('--output', default=None, help='Path to save data')
    parser.add_argument('--prefix', default=None, help='Dataset prefix')
    parser.add_argument('--ratio', type=float, default=0.2, help='Probability of remove nodes')
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()