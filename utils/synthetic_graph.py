import numpy as np
import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import json
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic graph.")
    parser.add_argument('--num_nodes', default=1000, help='Number of nodes in generated graph.', type=int)
    parser.add_argument('--min_degree', default=999, help="Min degree of a node.", type=int)
    parser.add_argument('--max_degree', default=999, help="Min degree of a node.", type=int)
    parser.add_argument('--feature_dim', default=50, help="Feature dimension.", type=int)
    parser.add_argument('--prefix', default="graph", help='Name the graph.')
    parser.add_argument('--out_dir', default="example_data/synthetic/", help="Output directory.")
    parser.add_argument('--seed', type=int, default=123, help='Seed of random generators')
    parser.add_argument('--visualize', type=bool, default=False, help="Whether visualize degrees or not.")
    return parser.parse_args()

def generate_graph(args):
    num_nodes = args.num_nodes
    while True:

        degrees = np.random.randint(args.min_degree, args.max_degree + 1, size = num_nodes)
        degrees = degrees.tolist()
        try:
            graph = nx.havel_hakimi_graph(degrees)
            print("Generated successfully.")
            break
        except Exception as err:
            print("Error. Generating graph...")
            continue

    if args.visualize:
        sorted_degs = sorted(degrees)
        plt.bar(np.arange(len(degrees)), sorted_degs)
        plt.xlabel('Index')
        plt.ylabel('Degree')
        plt.show()
    return graph

def init_features(shape):
    features = np.random.uniform(size=shape)
    for i, feat in enumerate(features):
        mask = np.ones(feat.shape, dtype=bool)
        mask[feat.argmax()] = False
        feat[~mask] = 1
        feat[mask] = 0
    return features


def save_graph(graph, args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.out_dir + "/dictionaries"):
        os.makedirs(args.out_dir+ "/dictionaries")
    if not os.path.exists(args.out_dir + "/edgelist"):
        os.makedirs(args.out_dir+ "/edgelist")
    if not os.path.exists(args.out_dir + "/graphsage"):
        os.makedirs(args.out_dir+"/graphsage")

    res = json_graph.node_link_data(graph)

    res['nodes'] = [
        {'id': str(node['id'])} for node in res['nodes']
    ]
    num_train = int(len(res['nodes'])*0.81)
    num_val = int(len(res['nodes'])*0.09)
    for idx, node in enumerate(res['nodes']):
        if idx <= num_train:
            node['val'] = False
            node['test'] = False
        elif idx <= num_train+num_val:
            node['val'] = True
            node['test'] = False
        else:
            node['val'] = False
            node['test'] = True

    res['links'] = [
            {
                'source': link['source'],
                'target': link['target']
            }
            for link in res['links']]

    id_map = {str(x['id']):i for i,x in enumerate(res['nodes'])}

    features = init_features((len(res['nodes']), args.feature_dim))

    with open(args.out_dir+"/graphsage/"+args.prefix + "-G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(args.out_dir+"/graphsage/"+args.prefix + "-id_map.json", 'w') as outfile:
        json.dump(id_map, outfile)
    np.save(args.out_dir+"/graphsage/"+args.prefix+"-feats.npy", features)

    nx.write_edgelist(graph, path=args.out_dir + "/edgelist/" + args.prefix + ".edgelist" , delimiter=" ", data=['weight'])
    with open(args.out_dir + "/dictionaries/groundtruth", 'w') as f:
        for node in res['nodes']:
            f.write("{0} {1}\n".format(node['id'], node['id']))
        f.close()
    print("Graph has been saved to {}".format(args.out_dir))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    np.random.seed(args.seed)
    graph = generate_graph(args)
    save_graph(graph, args)
