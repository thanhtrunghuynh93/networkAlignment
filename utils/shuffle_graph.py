import argparse
import numpy as np
import os
import pdb
import subprocess
import json
from edgelist_to_graphsage import edgelist_to_graphsage
from networkx.readwrite import json_graph
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle a graph and generate new file")
    parser.add_argument('--input_dir', default="/home/bigdata/thomas/dataspace/graph/ppi/sub_graph/subgraph3/random_delete_node/del,p=0.2",  help="Input directory")
    parser.add_argument('--out_dir', default="/home/bigdata/thomas/dataspace/graph/ppi/sub_graph/subgraph3/random_delete_node/del,p=0.2/permutation", help="Output directory")
    parser.add_argument('--seed', default=123, help="seed", type=int)
    return parser.parse_args()

def shuffle_graph(input_dir, output_dir):
    """
    :param input_dir: input directory that contains graphsage/ and /edgelist directory
    :param output_dir: output directory, 
    :return:
    """

def load_edgelist_file(nodes_file):    
    all_nodes = []
    all_edges = []
    if os.path.exists(nodes_file):
        with open(nodes_file) as fp:
            for line in fp:
                edge = line.split()
                if edge[0] not in all_nodes:
                    all_nodes.append(edge[0])
                if edge[1] not in all_nodes:
                    all_nodes.append(edge[1])                
                all_edges.append([edge[0], edge[1]])    
    all_nodes = np.array(all_nodes)
    return all_nodes, all_edges

def shuffle(all_nodes, all_edges, args):
    new_idxes = np.arange(0, len(all_nodes))
    np.random.shuffle(new_idxes)
    
    node_dict = {}
    node_rev_dict = {}
    permute_edges = []    
    permute_nodes = all_nodes[new_idxes]
    for i in range(len(all_nodes)):
        node_dict[all_nodes[i]] = permute_nodes[i]
        node_rev_dict[permute_nodes[i]] = all_nodes[i]

    for edge in all_edges:
        permute_edges.append([node_dict[edge[0]], node_dict[edge[1]]])

    np.random.shuffle(permute_edges)        
    return permute_edges, node_dict, node_rev_dict

def save(permute_edges, node_dict, node_rev_dict, input_dir, out_dir):

    assert (input_dir != out_dir), "Input and output must be different" 

    if not os.path.exists(out_dir + "/dictionaries"):
        os.makedirs(out_dir+ "/dictionaries")
    if not os.path.exists(out_dir + "/edgelist"):
        os.makedirs(out_dir+ "/edgelist")
    if not os.path.exists(out_dir + "/graphsage"):
        os.makedirs(out_dir+"/graphsage")

    with open(out_dir + "/dictionaries/groundtruth", 'w') as f:
        for key in node_dict.keys():
            f.write("{0} {1}\n".format(key, node_dict[key]))
        f.close()

    with open(out_dir + "/edgelist/edgelist", 'w') as f:
        for edge in permute_edges:
            f.write("{0} {1}\n".format(edge[0], edge[1]))
        f.close()

    # Call edgelist_to_graphsage
    print("Call edgelist to graphsage ")
    edgelist_to_graphsage(out_dir)
    old_idmap = json.load(open(input_dir + "/graphsage/" + "id2idx.json"))    
    new_idmap = json.load(open(out_dir + "/graphsage/" + "id2idx.json"))    
    new_nodes = list(new_idmap.keys())

    print("Saving new class map")
    class_map_file = Path(input_dir + "/graphsage/" + "class_map.json")
    if class_map_file.is_file():
        old_class_map = json.load(open(input_dir + "/graphsage/class_map.json")) # id to class
        new_class_map = {node: old_class_map[node_rev_dict[node]] for node in new_nodes}
        with open(out_dir + "/graphsage/" + "class_map.json", 'w') as outfile:
            json.dump(new_class_map, outfile)
    
    old_idxs = np.zeros(len(new_idmap.keys())).astype(int)
    
    for i in range(len(new_nodes)):
        new_id = new_nodes[i]
        old_id = node_rev_dict[new_id]
        old_idx = old_idmap[old_id]
        old_idxs[i] = old_idx

    print("Saving features")
    feature_file = input_dir + '/graphsage/' + 'feats.npy'
    features = None
    if os.path.isfile(feature_file):
        features = np.load(feature_file)
        features = features[old_idxs]
        np.save(out_dir + "/graphsage/" + "feats.npy", features)

    print("Dict and edgelist have been saved to: {0}".format(out_dir))

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    all_nodes, all_edges = load_edgelist_file(args.input_dir + "/edgelist/edgelist")
    permute_edges, node_dict, node_rev_dict = shuffle(all_nodes, all_edges, args)
    save(permute_edges, node_dict, node_rev_dict, args.input_dir, args.out_dir)

