import networkx as nx
import argparse
from input.fully_synthetic import FullySynthetic
import pdb
import numpy as np
import os
import json
from networkx.readwrite import json_graph

def parse_args():
    args = argparse.ArgumentParser(description="Create graph of n connected components.")
    args.add_argument("--n", default=1000, help="Number of nodes.", type=int)
    args.add_argument("--nc", default=2, help="Number of connected components.", type=int)
    args.add_argument("--feature_dim", default=50, help="Feature dimension.", type=int)
    args.add_argument("--k", default=10, type=int)
    args.add_argument("--p", default=0.5, type=float)
    args.add_argument("--output_dir",
                      default="../dataspace/graph/fully-synthetic/connected-components/smallworld-n1000-k10-p5-nc2",
                      help="Directory where new graph will be saved.")
    args.add_argument("--seed", default=100)
    return args.parse_args()

def random_number_of_nodes(n, nc, k):
    # find nc numbers, sum of them is equals to n
    arr = np.random.dirichlet(np.ones(nc)) * n
    arr = arr.astype(np.int16)
#     if value is less than n/10, increase it by 10
    arr[arr<(k+1)] += (k+1)
#     check sum arr
    if arr.sum() < n:
        arr[arr.argmin()] += n-arr.sum()
    elif arr.sum() > n:
        arr[arr.argmax()] -= arr.sum()-n
    return arr

def add_feature_to_graph(graph, features):
    if features is None:
        return
    for idx, node in enumerate(graph.nodes()):
        graph.node[node]['feature'] = features[idx].tolist()

def to_one_connected_components(graph):
    components = list(nx.connected_components(graph))
    while len(components) > 1:
        graph.add_edge(list(components[-2])[0], list(components[-1])[0])
        components = list(nx.connected_components(graph))

def relabel_graph(graph, from_new_idx):
    mapp = {}
    for idx, node in enumerate(graph.nodes()):
        mapp[node] = from_new_idx
        from_new_idx += 1
    graph2 = nx.relabel_nodes(graph, mapp)
    return graph2, from_new_idx

def check_graphs_labels_distinguish(graphs):
    labels = []
    total_nodes = 0
    set_nodes = set([])
    for graph in graphs:
        labels.append(graph.nodes())
        total_nodes += len(graph.nodes())
        set_nodes = set_nodes.union(set(graph.nodes()))
    assert total_nodes == len(set_nodes), "Graphs labels are not distinguish!"

def create_graph(n, nc, feature_dim, k, p, seed=100):
    """

    :param n: number of nodes
    :param nc: number of connected components
    :return: graph of networkx format
    """
    # np.random.seed(123)
    n_nodes = random_number_of_nodes(n, nc, k)
    graphs = []

    idx_node_large_graph = 0
    for n_node in n_nodes:
        graph, feature = FullySynthetic.generate_small_world_graph(
            None, n_node, k, p, feature_dim=feature_dim, seed=seed
        )
        add_feature_to_graph(graph, feature)
        print("====Subgraph===")
        print(nx.info(graph))
        to_one_connected_components(graph)
        graph, idx_node_large_graph = relabel_graph(graph, idx_node_large_graph)
        graphs.append(graph)

    check_graphs_labels_distinguish(graphs)

    while len(graphs) > 1:
        graph = nx.compose(graphs[0], graphs[1])
        graphs = [graph, *graphs[2:]]
    print("====Large graph====")
    print(nx.info(graphs[0]))
    print("Number of connected components:",
          len(list(nx.connected_components(graphs[0]))))
    return graphs[0]

def get_features_from_graph(graph):
    features = None
    if 'feature' in graph.node[graph.nodes()[0]]:
        features = []
        for idx, node in enumerate(graph.nodes()):
            features.append(graph.node[node]['feature'])
        features = np.array(features)
    return features

if __name__ == '__main__':
    args = parse_args()
    print(args)
    graph = create_graph(args.n, args.nc, args.feature_dim, args.k, args.p, args.seed)
    features = get_features_from_graph(graph)
    FullySynthetic.save_graph(args.output_dir, graph, features)
