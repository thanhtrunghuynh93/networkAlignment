import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import os
import json

class FullySynthetic():
    """
    this class provided these functions to generate fully synthetic graph
    - generate_havel_hakimi_graph: genarate synthetic graph by havel hakimi algorithm

    """
    def __init__(self):
        pass

    @staticmethod
    def generate_havel_hakimi_graph(output_dir, n_nodes, min_degree=1, max_degree=None,
                                    feature_dim=None):
        """
        :param output_dir: output directory to save graph as networkx format
        :param n_nodes: number of nodes of new graph
        :param min_degree: minimum degree of nodes
        :param max_degree: maximum degree of nodes, if None, max_degree = n_nodes - 1
        :param feature_dim: feature dimension, if None or is zero, not create feature
        :return:
        """
        np.random.seed(123)
        if max_degree is None:
            max_degree = n_nodes - 1
        degrees = np.random.randint(min_degree, max_degree+1,size=n_nodes)
        degrees = degrees.tolist()
        not_valid = 0
        while(not nx.is_valid_degree_sequence_havel_hakimi(degrees)):
            not_valid += 1
            if not_valid>=1000:
                raise Exception("1000 times random a not valid degrees. Stopped generating. "
                                "Try choose another min_degree and max_degree values.")
        graph = nx.havel_hakimi_graph(degrees)
        features = None
        if feature_dim is not None:
            assert type(feature_dim) == int, "feature_dim must be None or int type"
            features = FullySynthetic.random_features((n_nodes,feature_dim))
        if output_dir is not None:
            FullySynthetic.save_graph(output_dir, graph, features)
        return graph, features

    @staticmethod
    def generate_erdos_renyi_graph(output_dir, n_nodes, n_edges, seed=123, feature_dim=None):
        graph = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
        features = None
        if feature_dim is not None:
            assert type(feature_dim) == int, "feature_dim must be None or int type"
            features = FullySynthetic.random_features((n_nodes, feature_dim))
        if output_dir is not None:
            FullySynthetic.save_graph(output_dir, graph, features)
        return graph, features
    
    @staticmethod
    def generate_small_world_graph(output_dir, n_nodes, k_neighbors, p_edge_modify, seed=123, feature_dim=None):
        graph = nx.connected_watts_strogatz_graph(n=n_nodes, k=k_neighbors, p=p_edge_modify, seed=seed)
        features = None
        if feature_dim is not None:
            assert type(feature_dim) == int, "feature_dim must be None or int type"
            features = FullySynthetic.random_features((n_nodes, feature_dim))
        if output_dir is not None:
            FullySynthetic.save_graph(output_dir, graph, features)
        return graph, features

    @staticmethod
    def random_features(shape):
        features = np.random.uniform(size=shape)
        for i, feat in enumerate(features):
            mask = np.ones(feat.shape, dtype=bool)
            mask[feat.argmax()] = False
            feat[~mask] = 1
            feat[mask] = 0
        return features

    @staticmethod
    def save_graph(output_dir, graph, features=None):
        id2idx = {}
        for idx, node in enumerate(graph.nodes()):
            id2idx[node] = idx
        if not os.path.exists(output_dir+'/graphsage'):
            os.makedirs(output_dir+'/graphsage')
        if not os.path.exists(output_dir+'/edgelist'):
            os.makedirs(output_dir+'/edgelist')
        if not os.path.exists(output_dir+'/dictionaries'):
            os.makedirs(output_dir+'/dictionaries')
        nx.write_edgelist(graph, output_dir+"/edgelist/edgelist", delimiter=' ', data=False)
        with open(output_dir + '/graphsage/G.json', 'w+') as file:
            file.write(json.dumps(json_graph.node_link_data(graph)))
        with open(output_dir + '/graphsage/id2idx.json', 'w+') as file:
            file.write(json.dumps(id2idx))
        if features is not None:
            np.save(output_dir + '/graphsage/feats.npy', features)
            
        file = open(output_dir+"/dictionaries/groundtruth","w")         
        for idx, node in enumerate(graph.nodes()):            
            file.write("{0} {1}\n".format(node, node))
        file.close() 
        print("Graph has been saved to ", output_dir)