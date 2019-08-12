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

def read_dict(dict_file):

    all_nodes_source = set()
    all_nodes_target = set()
    all_instances = []
    if os.path.exists(dict_file):
        with open(dict_file) as fp:
            for line in fp:
                ins = line.split()
                all_instances.append([ins[0], ins[1]])
                all_nodes_source.add(ins[0])
                all_nodes_target.add(ins[1])
        
    return all_instances, all_nodes_source, all_nodes_target

def filter_data(nodes, dataset_dir, dataset_prefix, outdir):

    prefix = dataset_dir + "/" + dataset_prefix
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)    
    print(nx.info(G))

    outdir += "/" + dataset_prefix
    if not os.path.isdir(outdir): 
        os.makedirs(outdir)

    id_map = json.load(open(prefix + "-id_map.json"))    
        
    rem_nodes = []
    count_rm = 0   
    
    for node in G.nodes():    
        if node not in nodes:            
            G.remove_node(node)
            rem_nodes.append(node)
            if count_rm % 1000 == 0:
                print("\t{0}-th node removed:\t {1}".format(count_rm, node))            
            count_rm += 1

    print("Removed {0} nodes".format(len(rem_nodes)))
    print("Removing isolated nodes")
    for node in G.nodes():
        if G.degree(node) == 0:
            G.remove_node(node)
            rem_nodes.append(node)
            count_rm += 1
    
    print("Removed {0} nodes".format(len(rem_nodes)))

    data = json_graph.node_link_data(G)
    s = json.dumps(data,  indent=4, sort_keys=True)
    print(nx.info(G))

    edgelist_dir = outdir + "/" + dataset_prefix + ".edgelist"    
    nx.write_edgelist(G, path = edgelist_dir , delimiter=" ", data=['weight'])

    outdir += "/graphsage" 
    if not os.path.isdir(outdir): 
        os.makedirs(outdir)

    with open(outdir + "/" + dataset_prefix + "-G.json", 'w') as f:
        f.write(s)
        f.close()
        
    #Generate id_map files
    new_id_map = {}
    for i, node in enumerate(G.nodes()):
        new_id_map[str(node)] = i
    
    with open(outdir + "/" + dataset_prefix + "-id_map.json", 'w') as outfile:
        json.dump(new_id_map, outfile)
           
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
        new_feats = np.zeros((len(G.nodes()), feats.shape[1]))
        for node in G.nodes():
            old_idx = id_map[str(node)]
            new_idx = new_id_map[str(node)]
            new_feats[new_idx] = feats[old_idx]
        np.save(outdir + "/" + dataset_prefix + "-feats.npy",  new_feats)
    
    if os.path.exists(prefix + "-class_map.json"):
        copyfile(prefix + "-class_map.json", outdir + "/" + dataset_prefix + "-class_map.json")

    return G.nodes()

def filter_dictionary(all_instances, filtered_source_nodes, filtered_target_nodes, outdir):
    filtered_instances = []
    for instance in all_instances:
        if instance[0] in filtered_source_nodes and instance[1] in filtered_target_nodes:
            filtered_instances.append(instance)

    if not os.path.isdir(outdir + "/dictionaries"): 
        os.makedirs(outdir + "/dictionaries")

    with open(outdir + "/dictionaries/groundtruth", 'w') as f:
        for instance in filtered_instances:
            f.write("{0} {1}\n".format(instance[0], instance[1]))
        f.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--source_dataset_dir', default=None, help='Path to source dataset')
    parser.add_argument('--target_dataset_dir', default=None, help='Path to target dataset')
    parser.add_argument('--source_dataset_prefix', default=None, help='Source dataset prefix')
    parser.add_argument('--target_dataset_prefix', default=None, help='Target dataset prefix')
    parser.add_argument('--dict_file', default=None, help='Path to groundtruth file')
    parser.add_argument('--outdir', default=None, help='Path to save data')    
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    all_instances, all_nodes_source, all_nodes_target = read_dict(args.dict_file)    
    filtered_source_nodes = filter_data(all_nodes_source, args.source_dataset_dir, args.source_dataset_prefix, args.outdir)
    filtered_target_nodes = filter_data(all_nodes_target, args.target_dataset_dir, args.target_dataset_prefix, args.outdir)
    filter_dictionary(all_instances, filtered_source_nodes, filtered_target_nodes, args.outdir)
