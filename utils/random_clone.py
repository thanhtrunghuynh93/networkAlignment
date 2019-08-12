from __future__ import print_function, division

import networkx as nx
import numpy as np


def random_clone_synthetic(dataset, p_new_connection, p_remove_connection, seed):
    np.random.seed = seed
    H = dataset.G.copy()
    adj = dataset.get_adjacency_matrix()
    adj *= np.tri(*adj.shape)

    idx2id = {v: k for k,v in dataset.id2idx.items()}
    connected = np.argwhere(adj==1)

    mask_remove = np.random.uniform(0,1, size=(len(connected))) < p_remove_connection
    edges_remove = [(idx2id[x[0]], idx2id[x[1]]) for idx, x in enumerate(connected)
                    if mask_remove[idx] == True]
    count_rm = mask_remove.sum()
    H.remove_edges_from(edges_remove)

    print("New graph:")
    print("- Number of nodes:", len(H.nodes()))
    print("- Number of edges:", len(H.edges()))
    return H
