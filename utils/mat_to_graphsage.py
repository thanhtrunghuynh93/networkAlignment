from __future__ import print_function, division
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
from collections import defaultdict
import random
import scipy.io as sio
from scipy.sparse import coo_matrix
import argparse
import math
import sys
import csv
import pdb
import os
from scipy.io import savemat

# Author: TrungHT
# Created time: 18/6/2018

class MatLinkagePreprocessor:
    
    def __init__(self, mat_file_dir, dataset_name_1, dataset_name_2, output_dir):
        self.mat_file_dir = mat_file_dir
        self.dataset_name_1 = dataset_name_1
        self.dataset_name_2 = dataset_name_2
        self.output_dir = output_dir

    def process(self):

        dict = sio.loadmat(self.mat_file_dir)
        print("-------------Processing file at {0} and generating to GraphSAGE format and edgelist---------------".format(self.mat_file_dir))
        print("File specs:")

        print(dict.keys())

        key_list = []
        key_list.extend([self.dataset_name_1, self.dataset_name_1 + "_node_label", self.dataset_name_1 + "_edge_label"])
        key_list.extend([self.dataset_name_2, self.dataset_name_2 + "_node_label", self.dataset_name_2 + "_edge_label"])
        

        if "gndtruth" in dict.keys():
            key_list.extend(["gndtruth"])
            self.storeGroundTruth(dict["gndtruth"])
        if "ground_truth" in dict.keys():
            key_list.extend(["ground_truth"])
            self.storeGroundTruth(dict["ground_truth"])
        
        for key in key_list:
            print("{0} : {1} {2}".format(key, type(dict[key]), dict[key].shape))  
        print("---------------------------------------------------------------------------------")

        _, id2idx_src = self.processDataset(self.dataset_name_1,
                            dict[self.dataset_name_1], 
                            dict[self.dataset_name_1 + "_node_label"], 
                            dict[self.dataset_name_1 + "_edge_label"])

        _, id2idx_trg = self.processDataset(self.dataset_name_2,
                            dict[self.dataset_name_2], 
                            dict[self.dataset_name_2 + "_node_label"], 
                            dict[self.dataset_name_2 + "_edge_label"])

        if "H" in dict.keys():
            print(dict['H'].shape)
            key_list.extend(["H"])
            self.storeH(dict["H"], id2idx_src, id2idx_trg)
        
    def processDataset(self, dataset_name, adjacency_matrix, node_label, edge_label):

        print("Processing dataset {0}".format(dataset_name))
        dir = "{0}/{1}".format(self.output_dir,dataset_name)
        if not os.path.exists(dir + "/edgelist/"):
            os.makedirs(dir + "/edgelist/")
        dense_node_label = node_label.todense() # label can be 0, 1 or 2 | to one hot vector

        sources, targets = adjacency_matrix.nonzero() # source 0 -> 1117


        edgelist = zip(sources.tolist(), targets.tolist())
        G = nx.Graph(edgelist)
        edgelist_dir = dir + "/edgelist/edgelist"
        nx.write_edgelist(G, path = edgelist_dir , delimiter=" ", data=['weight'])

        print(nx.info(G))        
        print("Edgelist stored in {0}".format(edgelist_dir))

        # G = nx.read_edgelist(edgelist_dir)

        num_nodes = len(G.nodes())

        rand_indices = np.random.permutation(num_nodes)
        train = rand_indices[:int(num_nodes * 0.81)]

        val = rand_indices[int(num_nodes * 0.81):int(num_nodes * 0.9)]
        test = rand_indices[int(num_nodes * 0.9):]

        id_map = {}
        for i, node in enumerate(G.nodes()):
            id_map[str(node)] = i

        res = json_graph.node_link_data(G)

        res['nodes'] = [
            {
                'id': str(node['id']),
                'feature': np.squeeze(np.asarray(dense_node_label[int(node['id'])])).tolist(),
                'val': id_map[str(node['id'])] in val,
                'test': id_map[str(node['id'])] in test
            }
            for node in res['nodes']]
                    
        res['links'] = [
            {
                'source': link['source'],
                'target': link['target']
            }
            for link in res['links']]

        dir += "/graphsage/"
        if not os.path.exists(dir):
            os.makedirs(dir)
                       
        with open(dir + "G.json", 'w') as outfile:
            json.dump(res, outfile)       
        
        with open(dir + "id2idx.json", 'w') as outfile:
            json.dump(id_map, outfile)
        
        feats = np.zeros((dense_node_label.shape[0], dense_node_label.shape[1]))
        for id in id_map.keys():
            idx = id_map[id]
            feats[idx] = dense_node_label[int(id)]

        np.save(dir + "feats.npy",  feats)

        # save edge feats
        import scipy.sparse
        edge_feats = []
        for matrix in edge_label[0]:
            row_idxs, col_idxs = matrix.nonzero()
            values = matrix.toarray()[row_idxs,col_idxs]
            row_idxs = np.array([id_map[str(x)] for x in row_idxs])
            col_idxs = np.array([id_map[str(x)] for x in col_idxs])
            edge_feats.append(scipy.sparse.csc_matrix((values, (row_idxs, col_idxs)), shape=matrix.shape))

        savemat(dir+'/edge_feats.mat', {'edge_feats':edge_feats}, do_compression=True)

        print("GraphSAGE format stored in {0}".format(dir))
        print("----------------------------------------------------------")
        return G, id_map


    def storeH(self, H, id2idx_src, id2idx_trg):
        try:
            H = H.todense()
        except:
            pass
        new_H = np.zeros(H.shape)
        n_src_nodes = len(id2idx_src.keys())
        n_trg_nodes = len(id2idx_trg.keys())

        for i in range(n_src_nodes):
            for j in range(n_trg_nodes):
                new_H[id2idx_trg[str(j)], id2idx_src[str(i)]] = H[j, i]
        dict_H = {'H': new_H}
        savemat(self.output_dir+'/H.mat', dict_H, do_compression=True)
        print("Prior H have been saved to: ", self.output_dir + '/H.mat')

    def storeGroundTruth(self, groundTruth):
        dir = self.output_dir + "/dictionaries/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file = open(dir + "groundtruth","w") 
        
        for edge in groundTruth:
            #Matlab index count from 1
            file.write("{0} {1}\n".format(int(edge[0]) - 1, int(edge[1]) - 1)) 
        file.close() 

def parse_args():
    parser = argparse.ArgumentParser(description="Convert mat linkage data to dataset's edgelist and GraphSAGE format.")
    parser.add_argument('--input', default="../dataspace/graph/flickr_myspace/data.mat", help="Input data directory")
    parser.add_argument('--dataset1', default="flickr", help="Name of the dataset 1")
    parser.add_argument('--dataset2', default="myspace", help="Name of the dataset 2")
    parser.add_argument('--output', default="../dataspace/graph/flickr_myspace/", help="Input data directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    processor = MatLinkagePreprocessor(args.input, args.dataset1, args.dataset2, args.output)
    processor.process()

    # processor = MatLinkagePreprocessor("/home/trunght/workspace/dataset/graph/flickr_lastfm/data.mat", "flickr", "lastfm", "/home/trunght/workspace/dataset/graph/flickr_lastfm")
    # processor.process()
    
        
