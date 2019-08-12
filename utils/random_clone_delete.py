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

def add_and_remove_edges(G, p_remove_connection, num_remove=10):
    '''
    for each node,
      add a new connection to random other node, with prob p_new_connection,
      remove a connection, with prob p_remove_connection

    operates on G in-place
    '''
    rem_edges = []
    count_rm = 0
    for node in G.nodes():
        # find the other nodes this one is connected to
        # connected = [to for (fr, to) in G.edges(node)]
        connected = G.neighbors(node)

        # probabilistically remove a random edge
        if len(connected) > 1 and count_rm <= num_remove: # only try if an edge exists to remove
            if random.random() < p_remove_connection:
                count_rm += 1
                remove = random.choice(connected)
                if len(G.neighbors(remove)) > 1:
                    G.remove_edge(node, remove)
                    rem_edges.append( (node, remove) )

                    if count_rm % 1000 == 0:
                        print("\t{0}-th edge removed:\t {1} -- {2}".format(count_rm, node, remove))

        if count_rm > num_remove:
            break
    return rem_edges

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--input', default="/Users/tnguyen/dataspace/graph/ppi/graphsage", help='Path to load data')
    parser.add_argument('--output', default="/Users/tnguyen/dataspace/graph/ppi/random_clone", help='Path to save data')
    parser.add_argument('--prefix', default="ppi", help='Dataset prefix')
    parser.add_argument('--pdel', type=float, default=0.2, help='Probability of remove old edges')
    parser.add_argument('--ndel', type=float, default=0.2, help='Number of removed edges')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    return parser.parse_args()

def main(args):
    args.input += "/" 
    G_data = json.load(open(args.input + "G.json"))
    G = json_graph.node_link_graph(G_data)
    print(nx.info(G))

    H = G.copy()
    n = len(G.nodes())
    rem_edges = add_and_remove_edges(H, args.pdel, int(args.ndel * n))
    print("Remove {0} edges".format(len(rem_edges)))
    data = json_graph.node_link_data(H)
    s = json.dumps(data,  indent=4, sort_keys=True)
    print(nx.info(H))



    args.output += "/del_edge,p={0},n={1}".format(args.pdel, args.ndel)

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
