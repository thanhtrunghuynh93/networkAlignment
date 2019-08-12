import networkx as nx
import json
from networkx.readwrite import json_graph
import random
import os
import numpy as np
from input.dataset import Dataset
from utils.random_clone import random_clone_synthetic

class SemiSynthetic():
    """
    this class provided these functions:
    - generate_PALE_synthetic: generate 2 subgraphs with the algorithms mentioned in PALE paper.
    - generate_REGAL_synthetic: generate a graph with the algorithms mentioned in REGAL paper.
    - generate_random_clone_synthetic: generate a graph with probability of adding connection and probability of removing connection
    """
    def __init__(self, networkx_dir, output_dir1, output_dir2=None, groundtruth_dir=None, seed=1):
        """

        :param networkx_dir: directory contains graph data in networkx format
        :param output_dir1: output directory for subgraph1
        :param output_dir2: output directory for subgraph2
        """
        self.networkx_dir = networkx_dir
        self.output_dir1 = output_dir1
        self.output_dir2 = output_dir2
        self.groundtruth_dir = groundtruth_dir
        self.seed = seed
    
    def set_seed(self):
        random.seed = seed
        np.random.seed = seed

    def generate_PALE_synthetic(self, alpha_s=0.9, alpha_c=0.9):
        if self.output_dir2 is None:
            raise Exception("generate PALE requires output_dir2")
        print("Synthetic graph. alpha_s = {} and alpha_c = {}".format(alpha_s, alpha_c))
        G1, G2 = self._create_PALE_subnet(self.networkx_dir, alpha_s, alpha_c)
        self._save_graph(G1, self.output_dir1)
        self._save_graph(G2, self.output_dir2)
        print("Subgraph1 has been saved to ", self.output_dir1)
        print("Subgraph2 has been saved to ", self.output_dir2)

    def _create_PALE_subnet(self, networkx_dir, alpha_s, alpha_c):
        G_data = json.load(open(networkx_dir+"/G.json"))
        G = json_graph.node_link_graph(G_data)

        # source graph
        G1 = G.copy()
        # target graph
        G2 = G.copy()

        for edge in G.edges():
            remove_both = 1 - 2 * alpha_s + alpha_s * alpha_c
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

    def generate_random_clone_synthetic(self, p_new_connection, p_remove_connection, p_change_feats=None):
        print("===============")
        dataset = Dataset(self.networkx_dir)
        G = random_clone_synthetic(dataset, p_new_connection, p_remove_connection, self.seed)
        self._save_graph(G, self.output_dir1, p_change_feats)

    def generate_REGAL_synthetic(self):
        self.generate_random_clone_synthetic(0, 0.05)

    def _save_graph(self, G, output_dir, p_change_feats=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(output_dir+"/graphsage")
            os.makedirs(output_dir+"/edgelist")
        with open(os.path.join(output_dir,"graphsage/G.json"), "w+") as file:
            res = json_graph.node_link_data(G)
            file.write(json.dumps(res))
        with open(os.path.join(output_dir,"graphsage/id2idx.json"), "w+") as file:
            file.write(json.dumps(self._create_id2idx(G)))
        features = self._build_features(p_change_feats)
        if features is not None:
            np.save(os.path.join(output_dir,"graphsage/feats.npy"), features)
        nx.write_edgelist(G, os.path.join(output_dir,"edgelist/edgelist"), delimiter=' ', data=False)
        print("Graph has been saved to ", self.output_dir1)

    def _create_id2idx(self, G):
        id2idx = {}
        for idx, node in enumerate(G.nodes()):
            id2idx[node] = idx
        return id2idx

    def _build_features(self, p_change_feats=None):
        features = None
        if os.path.isfile(self.networkx_dir+"/feats.npy"):
            features_ori = np.load(self.networkx_dir+"/feats.npy")
            if p_change_feats is not None:
                classes = np.unique(features_ori, axis=0)
                mask = np.random.uniform(size=(features_ori.shape[0]))
                mask = mask <= p_change_feats
                indexes_choice = np.random.choice(np.arange(classes.shape[0]),
                                                      size=(mask.sum()), replace=True)
                features_ori[mask] = classes[indexes_choice]
            features = features_ori
        return features
