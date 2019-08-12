import argparse
import numpy as np
import os
import pdb
import subprocess
import json
from edgelist_to_graphsage import edgelist_to_graphsage
from networkx.readwrite import json_graph
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Count number of pairs neighbor-nodes have same features.")
    parser.add_argument('--prefix', default="karate", help="prefix")
    return parser.parse_args()

def load_data(prefix):
    G_data = json.loads(open(prefix+"-G.json", "r").read())
    G = json_graph.node_link_graph(G_data)

    features = np.load(prefix+"-feats.npy")

    id_map = json.loads(open(prefix+"-id_map.json", "r").read())

    print("Number of nodes: ", len(G.nodes()))
    print("Number of edges: ", len(G.edges()))
    return G, features, id_map

def count_neighbor_nodes_same_features(G, features, id_map):
    count = 0
    feats_dim = features.shape[1]
    for node in G.nodes():
        neibs = G.neighbors(node)
        for neib in neibs:
            if (features[id_map[str(node)]] == features[id_map[str(neib)]]).sum() == feats_dim:
                count += 1

    n_pairs = count//2
    print("Number of pairs neighbor-nodes have same features: ", n_pairs, "(base on number of edges)")
    return n_pairs

if __name__ == '__main__':
    args = parse_args()
    G, features, id_map = load_data(args.prefix)
    count_neighbor_nodes_same_features(G, features, id_map)
