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

def read_data(dataset_dir, dataset_prefix):

    prefix = dataset_dir + "/" + dataset_prefix
    id_map = json.load(open(prefix + "-id_map.json"))            
    feats = np.load(prefix + "-feats.npy")
    return id_map, feats

def check_dict_features(all_instances, source_id_map, source_feats, target_id_map, target_feats):
    correct_case = 0
    for instance in all_instances:        
        source = np.argmax(source_feats[source_id_map[instance[0]]])
        target = np.argmax(target_feats[target_id_map[instance[1]]])
        if source == target:
            correct_case += 1
    
    print("Accuracy = {0}".format(correct_case / len(all_instances)))

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--source_dataset_dir', default=None, help='Path to source dataset')
    parser.add_argument('--target_dataset_dir', default=None, help='Path to target dataset')
    parser.add_argument('--source_dataset_prefix', default=None, help='Source dataset prefix')
    parser.add_argument('--target_dataset_prefix', default=None, help='Target dataset prefix')
    parser.add_argument('--dict_file', default=None, help='Path to groundtruth file')
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    all_instances, all_nodes_source, all_nodes_target = read_dict(args.dict_file) 
    source_id_map, source_feats = read_data(args.source_dataset_dir, args.source_dataset_prefix)
    target_id_map, target_feats = read_data(args.target_dataset_dir, args.target_dataset_prefix)
    check_dict_features(all_instances, source_id_map, source_feats, target_id_map, target_feats)
