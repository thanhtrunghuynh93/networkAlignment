from __future__ import print_function, division
import numpy as np
import random
import json
import sys
import os
import argparse
import os
import networkx as nx
from networkx.readwrite import json_graph
import pdb

def create_dictionary(filename, node_ids, s_prefix="", t_prefix=""):
    with open(filename, 'wt') as f:
        for i in node_ids:
            f.write("%s%s %s%s\n" % (s_prefix, i, t_prefix, i))
    f.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dict.")
    parser.add_argument('--input', default="/Users/tnguyen/dataspace/graph/ppi/graphsage/ppi-G.json", help='Path to load dataset')
    parser.add_argument('--dict', default="/Users/tnguyen/dataspace/graph/ppi/dictionaries/", help='Path to save dictionaries')
    parser.add_argument('--sprefix', default="", help='Source prefix')
    parser.add_argument('--tprefix', default="", help='Target prefix')
    parser.add_argument('--split', type=float, default=0.2, help='Train/test split')
    parser.add_argument('--seed', type=int, default=123, help='Seed of random generators')
    return parser.parse_args()

def main(args):
    G_data = json.load(open(args.input))
    G = json_graph.node_link_graph(G_data)
    n = len(G.nodes())
    node_ids = np.random.permutation(G.nodes())
    
    if not os.path.exists(args.dict):
        os.makedirs(args.dict)

    # node_ids = range(n)
    create_dictionary("{0}/node,split={1}.full.dict".format(args.dict, args.split), node_ids, args.sprefix, args.tprefix)
    create_dictionary("{0}/node,split={1}.train.dict".format(args.dict, args.split), node_ids[0:int(args.split * n)], args.sprefix, args.tprefix)
    create_dictionary("{0}/node,split={1}.test.dict".format(args.dict, args.split), node_ids[int(args.split * n):n], args.sprefix, args.tprefix)
    return

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)

