from __future__ import print_function, division
import numpy as np
import random
import json
import os
import argparse
import networkx as nx
from networkx.readwrite import json_graph
import pdb



def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--input', default='../dataspace/graph/ppi/subgraphs/subgraph3/',
                        help='Path to load data')
    parser.add_argument('--ratio', type=float, default=0.1, help='Probability of remove nodes')
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()



def load_data(data_dir):
    G_data = json.load(open(args.input + "/graphsage/G.json"))
    G = json_graph.node_link_graph(G_data)

    id2idx = json.load(open(data_dir + "/graphsage/id2idx.json"))
    feats = None
    if os.path.isfile(data_dir + "/graphsage/feats.npy"):
        feats = np.load(data_dir + "/graphsage/feats.npy")
    return G, id2idx, feats


def remove_nodes(data_dir, p_remove, seed):
    '''
    for each node,
    remove with prob p
    operates on G in-place
    '''
    G, id2idx, feats = load_data(data_dir)
    count_rm = 0
    for node in G.nodes():
        # probabilistically remove a random node
        if random.random() < p_remove:
            G.remove_node(node)
            if (count_rm + 1) % 1000 == 0:
                print("\t{}-th node removed:\t {}".format(count_rm, node))
            count_rm += 1
    # Remove isolated node
    for node in G.nodes():
        if G.degree(node) == 0:
            G.remove_node(node)
            count_rm += 1
    print("Remove total {0} nodes".format(count_rm))

    # Create new id2idx
    new_G = G
    new_id2idx = {}
    new_feats = None

    for idx, node in enumerate(new_G.nodes()):
        new_id2idx[node] = idx
    if feats is not None:
        new_feats = np.zeros((len(new_G.nodes()), feats.shape[1]))
        for node in G.nodes():
            new_feats[new_id2idx[node]] = feats[id2idx[node]]

    output_dir = data_dir + "/del-nodes-p{}-seed{}".format(str(p_remove).replace("0.", ""), seed)
    save_graph(output_dir, new_G, new_id2idx, new_feats)


def save_graph(output, G, id2idx, feats):
    if not os.path.exists(output + "/graphsage"):
        os.makedirs(output + "/graphsage/")
    if not os.path.exists(output + "/edgelist"):
        os.makedirs(output + "/edgelist")
    with open(output + "/graphsage/G.json", "w+") as file:
        file.write(json.dumps(json_graph.node_link_data(G)))
    with open(output + "/graphsage/id2idx.json", "w+") as file:
        file.write(json.dumps(id2idx))
    if feats is not None:
        np.save(output + "/graphsage/feats.npy", feats)
    nx.write_edgelist(G, output + "/edgelist/edgelist", delimiter=" ")
    print("New graph has been saved to ", output)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    remove_nodes(args.input, args.ratio, args.seed)
