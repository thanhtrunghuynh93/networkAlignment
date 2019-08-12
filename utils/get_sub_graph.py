from __future__ import print_function, division
from pathlib import Path
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate subgraphs of a network.")
    parser.add_argument('--input', default="../dataspace/graph/fq-tw-data/foursquare", help='Path to load data')
    parser.add_argument('--output', default="../dataspace/graph/fq-tw-data/foursquare/subgraphs", help='Path to save data')
    parser.add_argument('--prefix', default="ppi", help='Dataset prefix')
    parser.add_argument('--min_node', type=int, default=100, help='minimum node for subgraph to be kept')
    return parser.parse_args()

def main(args):
    G_data = json.load(open(args.input + "/graphsage/" + "G.json"))
    G = json_graph.node_link_graph(G_data)

    if isinstance(G.nodes()[0], int):
        def conversion(n): return int(n)
    else:
        def conversion(n): return n

    mapping = {conversion(G.nodes()[i]):str(G.nodes()[i]) for i in range(len(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    print("Original graph info: ")
    print(nx.info(G))

    print("Start extracting sub graph")
    max_num_nodes = 0
    all_subgraphs = list(nx.connected_component_subgraphs(G))
    subgraphs = []
    for graph in all_subgraphs:
        if len(graph.nodes()) > args.min_node:
            subgraphs.append(graph)

    i = 0
    for G in subgraphs:
        save_new_graph(G, args.input, args.output + "/subgraph" + str(i) + "/" , args.prefix)
        i += 1

    return

def save_new_graph(G, input_dir, output_dir, prefix):
    nodes = G.nodes()
    if not os.path.exists(output_dir+ "/edgelist/"):
        os.makedirs(output_dir+ "/edgelist/")
    if not os.path.exists(output_dir+ "/graphsage/"):
        os.makedirs(output_dir+ "/graphsage/")

    nx.write_edgelist(G, path = output_dir + "/edgelist/" + ".edgelist" , delimiter=" ", data=['weight'])

    output_prefix = output_dir + "/graphsage/"
    print("Saving new class map")
    input_dir += "/graphsage/"
    id2idx_file = Path(input_dir + "id2idx.json")
    if id2idx_file.is_file():
        id2idx = json.load(open(input_dir + "id2idx.json")) # id to class
        new_id2idx = {node: id2idx[node] for node in nodes}
        with open(output_prefix + 'id2idx.json', 'w') as outfile:
            json.dump(new_id2idx, outfile)
    print("Saving new id map")
    new_idmap = {node: i for i, node in enumerate(nodes)}
    with open(output_prefix + 'id2idx.json', 'w') as outfile:
        json.dump(new_idmap, outfile)

    print("Saving features")
    old_idmap = json.load(open(input_dir + "id2idx.json"))
    feature_file = Path(input_dir + 'feats.npy')
    features = None
    if feature_file.is_file():
        features = np.load(feature_file)
        new_idxs = np.zeros(len(nodes)).astype(int)
        for node in nodes:
            new_idx = new_idmap[node]
            old_idx = old_idmap[node]
            new_idxs[new_idx] = old_idx

        features = features[new_idxs]
        np.save(output_prefix + "feats.npy", features)

    print("Saving new graph")
    num_nodes = len(G.nodes())
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]

    id2idx = new_idmap
    res = json_graph.node_link_data(G)
    res['nodes'] = [
            {
                'id': str(node['id']),
                'val': id2idx[str(node['id'])] in val,
                'test': id2idx[str(node['id'])] in test
            }
            for node in res['nodes']]

    res['links'] = [
            {
                'source': link['source'],
                'target': link['target']
            }
            for link in res['links']]

    with open(output_prefix + "G.json", 'w') as outfile:
        json.dump(res, outfile)

    print("DONE!")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    main(args)