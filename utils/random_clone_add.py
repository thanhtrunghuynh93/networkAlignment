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

def add_and_remove_edges(G, p_new_connection, num_add=10):
    '''
    for each node,
      add a new connection to random other node, with prob p_new_connection,
      remove a connection, with prob p_remove_connection

    operates on G in-place
    '''
    new_edges = []
    rem_edges = []
    count_rm = 0
    count_add = 0
    for node in G.nodes():
        # find the other nodes this one is connected to
        # connected = [to for (fr, to) in G.edges(node)]
        connected = G.neighbors(node)
        # and find the remainder of nodes, which are candidates for new edges
        # unconnected = [n for n in G.nodes() if not n in connected]
        unconnected = [n for n in nx.non_neighbors(G, node)]

        # probabilistically add a random edge
        if len(unconnected) and count_add <= num_add: # only try if new edge is possible
            if random.random() < p_new_connection:
                count_add += 1
                new = random.choice(unconnected)
                G.add_edge(node, new)
                new_edges.append( (node, new) )
                # book-keeping, in case both add and remove done in same cycle
                # unconnected.remove(new)
                # connected.append(new)
                if count_add % 1000 == 0:
                    print("\t{0}-th new edge:\t {1} -- {2}".format(count_add, node, new))

        if count_add > num_add:
            break
    return rem_edges, new_edges

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--input', default="/Users/tnguyen/dataspace/graph/ppi/graphsage/", help='Path to load data')
    parser.add_argument('--output', default="/Users/tnguyen/dataspace/graph/ppi/random_clone/", help='Path to save data')
    parser.add_argument('--prefix', default="ppi", help='Dataset prefix')
    parser.add_argument('--padd', type=float, default=0.2, help='Probability of adding new edges')
    parser.add_argument('--nadd', type=float, default=0.2, help='Number of added edges')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    return parser.parse_args() 

def main(args):
    args.input += "/"
    G_data = json.load(open(args.input + "G.json"))
    G = json_graph.node_link_graph(G_data)
    print(nx.info(G))

    H = G.copy()
    n = len(G.nodes())
    rem_edges, new_edges = add_and_remove_edges(H, args.padd, int(args.nadd * n))
    print("Remove {0} and add {1} edges".format(len(rem_edges), len(new_edges)))
    data = json_graph.node_link_data(H)
    s = json.dumps(data,  indent=4, sort_keys=True)
    print(nx.info(H))

    args.output += "/add_edge,p={0},n={1}".format(args.padd, args.nadd)

    if not os.path.isdir(args.output):
        os.makedirs(args.output+'/edgelist')
        os.makedirs(args.output+'/graphsage')
        os.makedirs(args.output+'/dictionaries')

    edgelist_dir = args.output + "/edgelist/" + args.prefix + ".edgelist"
    if not os.path.isdir(args.output): os.makedirs(args.output)
    nx.write_edgelist(H, path = edgelist_dir , delimiter=" ", data=['weight'])
    args.output += "/graphsage/"
    with open(args.output + "G.json", 'w') as f:
        f.write(s)
        f.close()

    copyfile(args.input + "id2idx.json", args.output + "id2idx.json")
    if os.path.exists(args.input + "class_map.json"):
        copyfile(args.input + "class_map.json", args.output + "class_map.json")
    if os.path.exists(args.input + "feats.npy"):
        copyfile(args.input + "feats.npy", args.output + "feats.npy")
    # if os.path.exists(args.input + "-walks.txt"):
    #     copyfile(args.input + "-walks.txt", args.output + "-walks.txt")

    return

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)