import argparse
import json
import os
import time
import numpy as np

import networkx as nx
from networkx.readwrite import json_graph

try:
    import cPickle as pickle
except ImportError:
    import pickle

from algorithms.REGAL.xnetmf import *
from algorithms.REGAL.models import *
from algorithms.REGAL.alignments import *

from algorithms.network_alignment_model import NetworkAlignmentModel
from input.dataset import Dataset
import pdb

class REGAL(NetworkAlignmentModel):

    def __init__(self, source_dataset, target_dataset, max_layer=2, alpha=0.01, k=10, num_buckets=2,
                      gammastruc=1, gammaattr=1, normalize=True, num_top=None):                      
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_emb = None
        self.target_emb = None
        self.alignment_matrix = None
        self.max_layer = max_layer
        self.alpha = alpha 
        self.k = k
        self.num_buckets = num_buckets
        self.gammastruc = gammastruc
        self.gammaattr = gammaattr
        self.normalize = normalize
        self.num_top = num_top

    def merge_source_target_graphs(self):
        
        G1 = self.source_dataset.G
        G2 = self.target_dataset.G
        res1 = json_graph.node_link_data(G1)
        res2 = json_graph.node_link_data(G2)
        id2idx1 = self.source_dataset.id2idx
        id2idx2 = self.target_dataset.id2idx

        # Merging source graph and target graph into a big graph
        # id of source nodes equal to str(index_of_that_node_in_source_graph)
        # id of target nodes equal to str(index_of_that_node_in_target_graph + num_source_nodes)
        # index of source nodes equal to (index_of_that_node_in_source_graph)
        # index of target nodes equal to (index_of_that_node_in_target_graph + num_source_nodes)

        new_nodes_idxs1 = np.arange(len(G1.nodes()))
        new_nodes_idxs2 = np.arange(len(G1.nodes()), len(G1.nodes()) + len(G2.nodes()))
        
        new_nodes = []
        for idx, node in enumerate(res1["nodes"]):
            original_index = id2idx1[node["id"]]
            node["id"] = str(original_index)
            new_nodes.append(node)
        for idx, node in enumerate(res2["nodes"]):
            original_index = id2idx2[node["id"]]
            node["id"] = str(int(original_index) + len(G1.nodes()))
            new_nodes.append(node)
        
        new_id2idx = {}
        for node in new_nodes:
            new_id2idx[node["id"]] = int(node["id"])
        
        new_links = []
        for link in res1["links"]:
            new_source_index = link["source"]
            new_target_index = link["target"]
            new_links.append({
                'source': new_source_index,
                'target': new_target_index
            })
        for link in res2["links"]:
            new_source_index = link["source"] + len(G1.nodes())
            new_target_index = link["target"] + len(G1.nodes())
            new_links.append({
                'source': new_source_index,
                'target': new_target_index
            })

        new_features = None
        features1 = self.source_dataset.features
        features2 = self.target_dataset.features

        if features1 is not None and features2 is not None:
            if features1.shape[1] != features2.shape[1]:
                print("Can not create new features due to different features shape.")
            new_features = np.zeros((features1.shape[0] + features2.shape[0], features1.shape[1]))
            for i, feat in enumerate(features1):
                new_features[i] = feat
            for i, feat in enumerate(features2):
                new_features[i+len(G1.nodes())] = feat

        new_res = json_graph.node_link_data(G1)
        new_res["nodes"] = new_nodes
        new_res["links"] = new_links

        G = json_graph.node_link_graph(new_res)  

        return G, new_id2idx, new_features, new_nodes_idxs1, new_nodes_idxs2

    def align(self):
        G, id2idx, feats, src_idxs, trg_idxs = self.merge_source_target_graphs()
        adj = nx.adjacency_matrix(G)
        print ("learning representations...")
        before_rep = time.time()
        embed = self.learn_representations(adj, feats)
        after_rep = time.time()
        print("Learned representations in %f seconds" % (after_rep - before_rep))

        self.source_emb, self.target_emb = embed[src_idxs], embed[trg_idxs]

        before_align = time.time()        
        self.alignment_matrix = get_embedding_similarities(self.source_emb, self.target_emb, self.num_top)
        
        # Report scoring and timing
        after_align = time.time()
        total_time = after_align - before_align
        print("Align time in %f seconds" % total_time)
        return self.alignment_matrix

    def learn_representations(self, adj, feats):
        graph = Graph(adj = adj, node_attributes = feats)        
        rep_method = RepMethod(max_layer=self.max_layer, alpha=self.alpha, k=self.k, num_buckets=self.num_buckets, normalize=self.normalize, gammastruc=self.gammastruc, gammaattr=self.gammaattr)        
        
        print("Learning representations with max layer %d and alpha = %f" % (self.max_layer, self.alpha))
        representations = get_representations(graph, rep_method)
        # representations.dump(args.output)
        return representations

    def get_alignment_matrix(self):
        return self.alignment_matrix

    def get_source_embedding(self):
        return self.source_emb	

    def get_target_embedding(self):
        return self.target_emb

def parse_args():
    parser = argparse.ArgumentParser(description="Run REGAL.")
    parser.add_argument('--prefix1',             default="/home/trunght/dataspace/graph/douban/offline/graphsage/")
    parser.add_argument('--prefix2',             default="/home/trunght/dataspace/graph/douban/online/graphsage/")
    parser.add_argument('--groundtruth',
                        default='../../../dataspace/graph/douban/dictionaries/groundtruth')
    parser.add_argument('--output', nargs='?', default='emb/douban.emb',
                        help='Embeddings path')

    # parser.add_argument('--attributes', nargs='?', default=None,
    #                     help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

    parser.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser.add_argument('--max_layer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default=0.01, help="Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser.add_argument('--num_top', type=int, default=10,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    return parser.parse_args()

def main(args):

    source_dataset = Dataset(args.prefix1)
    target_dataset = Dataset(args.prefix2)

    model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    S = model.align()

if __name__ == "__main__":
    args = parse_args()
    main(args)

# def merge_graphs(args):
#     res, id2idx, feats, src_ids, trg_ids, gt = generate_graphs(args.prefix1, args.prefix2, args.groundtruth)

#     src_idxs = [id2idx[x] for x in src_ids]
#     trg_idxs = [id2idx[x] for x in trg_ids]
#     G = json_graph.node_link_graph(res)
#     #     convert groundtruth from id-one-graph-based to index-two-graphs-based
#     trg_idxgraph2idxarray = {}
#     for idx, trg_idx in enumerate(trg_idxs):
#         trg_idxgraph2idxarray[trg_idx] = idx
#     gt = {id2idx[k]: trg_idxgraph2idxarray[id2idx[v]] for k, v in gt.items()}
#     adj = nx.adjacency_matrix(G)
#     return adj, gt, feats, src_idxs, trg_idxs

# def load_graph(prefix):
    #     G_data = json.load(open(prefix + "-G.json"))
    #     G = json_graph.node_link_graph(G_data)
    #     res = json_graph.node_link_data(G)

    #     id_map = json.load(open(prefix + "-id_map.json"))

    #     features = None
    #     if os.path.isfile(prefix + "-feats.npy"):
    #         features = np.load(prefix + "-feats.npy")
    #     return G, id_map, res, features

# def main1(args):
#     adj, true_alignments, feats, src_idxs, trg_idxs = merge_graphs(args)
#     # Load in attributes if desired (assumes they are numpy array)
#     if args.attributes is not None:
#         args.attributes = np.load(args.attributes)  # load vector of attributes in from file
#         print (args.attributes.shape)

#     # Learn embeddings and save to output
#     print ("learning representations...")
#     before_rep = time.time()
#     embed = learn_representations(args, adj)
#     after_rep = time.time()
#     print("Learned representations in %f seconds" % (after_rep - before_rep))

#     # Score alignments learned from embeddings
#     # embed = np.load(args.output)
#     emb1, emb2 = embed[src_idxs], embed[trg_idxs]
#     before_align = time.time()
#     if args.numtop == 0:
#         args.numtop = None
#     alignment_matrix = get_embedding_similarities(emb1, emb2, num_top=args.numtop)

#     # Report scoring and timing
#     after_align = time.time()
#     total_time = after_align - before_align
#     print("Align time: "), total_time

#     if true_alignments is not None:
#         topk_scores = [1, 5, 10, 20, 50]
#         for k in topk_scores:
#             score, correct_nodes = score_alignment_matrix(alignment_matrix, topk=k, true_alignments=true_alignments)
#             print("score top%d: %f" % (k, score))


# Should take in a file with the input graph as edgelist (args.input)
# Should save representations to args.output

# def build_idmap(graph):
#     idmap = {}
#     for idx, node in enumerate(graph.nodes()):
#         idmap[node] = idx
#     return idmap

