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


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--input', default="$HOME/dataspace/graph/pale_facebook/graphsage/")
    parser.add_argument('--output1', default="$HOME/dataspace/graph/pale_facebook/random_clone/")
    parser.add_argument('--output2', default="$HOME/dataspace/graph/pale_facebook/random_clone/")
    parser.add_argument('--prefix', default="pale_facebook", help='Dataset prefix')
    parser.add_argument('--alpha_c', type=float, default=0.9)
    parser.add_argument('--alpha_s', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    return parser.parse_args()

def create_subnet(G, alpha_s, alpha_c):
    # source graph
    G1 = G.copy()
    # target graph
    G2 = G.copy()

    for edge in G.edges():
        remove_both = 1 - 2*alpha_s + alpha_s*alpha_c
        keep_source = 1 - alpha_s
        keep_target = 1 - alpha_s * alpha_c

        p = random.random()

        if p <= remove_both:
            # only remove edge of nodes have more than 2 neighbors
            if len(G1.neighbors(edge[0])) > 1 and len(G1.neighbors(edge[1])) > 1 \
                    and len(G2.neighbors(edge[0])) > 1 and len(G2.neighbors(edge[1])) > 1:
                G1.remove_edge(edge[0], edge[1])
                G2.remove_edge(edge[0], edge[1])
        elif p <= keep_source:
            if len(G2.neighbors(edge[0])) > 1 and len(G2.neighbors(edge[1])) > 1:
                G2.remove_edge(edge[0], edge[1])
        elif p <= keep_target:
            if len(G1.neighbors(edge[0])) > 1 and len(G1.neighbors(edge[1])) > 1:
                G1.remove_edge(edge[0], edge[1])
    print("Number of original edges is: ", len(G.edges()))
    print("Number of source edges is: ", len(G1.edges()))
    print("Number of target edges is: ", len(G2.edges()))

    return G1, G2


def main(args):
    args.input += "/" + args.prefix
    G_data = json.load(open(args.input + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    print(nx.info(G))

    H = G.copy()
    G1, G2 = create_subnet(H, args.alpha_s, args.alpha_c)
    data1 = json_graph.node_link_data(G1)
    data2 = json_graph.node_link_data(G2)
    s1 = json.dumps(data1,  indent=4, sort_keys=True)
    s2 = json.dumps(data2, indent=4, sort_keys=True)
    print("About G1")
    print(nx.info(G1))
    print("About G2")
    print(nx.info(G2))


    args.output1 += "sourceclone,alpha_c={0},alpha_s={1}".format(args.alpha_c, args.alpha_s)
    args.output2 += "targetclone,alpha_c={0},alpha_s={1}".format(args.alpha_c, args.alpha_s)

    edgelist_dir1 = args.output1 + "/" + args.prefix + ".edgelist"
    edgelist_dir2 = args.output2 + "/" + args.prefix + ".edgelist"
    if not os.path.isdir(args.output1): os.makedirs(args.output1)
    if not os.path.isdir(args.output2): os.makedirs(args.output2)

    nx.write_edgelist(G1, path = edgelist_dir1 , delimiter=" ", data=['weight'])
    nx.write_edgelist(G2, path = edgelist_dir2 , delimiter=" ", data=['weight'])

    args.output1 += "/" + args.prefix
    args.output2 += "/" + args.prefix
    with open(args.output1 + "-G.json", 'w') as f:
        f.write(s1)
        f.close()

    with open(args.output2 + "-G.json", 'w') as f:
        f.write(s2)
        f.close()


    copyfile(args.input + "-id_map.json", args.output1 + "-id_map.json")
    copyfile(args.input + "-id_map.json", args.output2 + "-id_map.json")
    if os.path.exists(args.input + "-class_map.json"):
        copyfile(args.input + "-class_map.json", args.output1 + "-class_map.json")
        copyfile(args.input + "-class_map.json", args.output2 + "-class_map.json")

    if os.path.exists(args.input + "-feats.npy"):
        copyfile(args.input + "-feats.npy", args.output1 + "-feats.npy")
        copyfile(args.input + "-feats.npy", args.output2 + "-feats.npy")

    if os.path.exists(args.input + "-walks.txt"):
        copyfile(args.input + "-walks.txt", args.output1 + "-walks.txt")
        copyfile(args.input + "-walks.txt", args.output2 + "-walks.txt")


    return

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)