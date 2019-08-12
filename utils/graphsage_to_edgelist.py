import argparse
import numpy as np
import os
import pdb
import networkx as nx 
from networkx.readwrite import json_graph
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert data from graphsage format to edgelist format.")
    parser.add_argument('--input_dir',  default=None, help="Dataset prefix.")
    parser.add_argument('--out_dir', default=None, help="Output prefix.")
    parser.add_argument('--prefix', default=None, help="Output prefix.")
    return parser.parse_args()

def convert(args):
    G_data = json.load(open(args.input_dir + "/graphsage/" + args.prefix +"-G.json", "r"))
    G = json_graph.node_link_graph(G_data)
    if not os.path.exists(args.out_dir+ "/edgelist/"):
        os.makedirs(args.out_dir+ "/edgelist/")    
    nx.write_edgelist(G, path = args.out_dir + "/edgelist/" + args.prefix + ".edgelist" , delimiter=" ", data=['weight'])
    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    convert(args)
