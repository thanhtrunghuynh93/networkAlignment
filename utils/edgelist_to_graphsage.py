from collections import defaultdict
import argparse
import numpy as np
import json
import os
import networkx as nx
from networkx.readwrite import json_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Generate graphsage format from edgelist")
    parser.add_argument('--out_dir', default=None, help="Output directory")
    parser.add_argument('--prefix', default="karate", help="seed")
    parser.add_argument('--seed', default=121, type=int)
    return parser.parse_args()

def edgelist_to_graphsage(dir, seed=121):
    np.random.seed(seed)
    edgelist_dir = dir + "/edgelist/edgelist"
    print(edgelist_dir)
    G = nx.read_edgelist(edgelist_dir)
    print(nx.info(G))
    num_nodes = len(G.nodes())
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.81)]
    val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
    test = rand_indices[int(num_nodes * 0.9):]

    id2idx = {}
    for i, node in enumerate(G.nodes()):
        id2idx[str(node)] = i

    res = json_graph.node_link_data(G)
    res['nodes'] = [
        {
            'id': node['id'],
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

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")

    with open(dir + "/graphsage/" + "G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + "id2idx.json", 'w') as outfile:
        json.dump(id2idx, outfile)

    print("GraphSAGE format stored in {0}".format(dir + "/graphsage/"))
    print("----------------------------------------------------------")

if __name__ == "__main__":
    args = parse_args()
    datadir = args.out_dir
    dataset = args.prefix
    edgelist_to_graphsage(datadir)



