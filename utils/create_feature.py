import argparse
import numpy as np


from input.dataset import Dataset



def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic feature for graph")
    parser.add_argument('--input_data1', default="")
    parser.add_argument('--input_data2', default="")
    parser.add_argument('--feature_dim', default=128)
    return parser.parse_args()


def create_features(data1, data2, dim):
    feature1 = create_feature(data1, dim)
    feature2 = create_feature(data2, dim)
    return feature1, feature2

def create_feature(data, dim):
    deg = data.get_nodes_degrees()
    deg = np.array(deg)
    binn = int(max(deg) / dim)
    feature = np.zeros((len(data.G.nodes()), dim))

    for i in range(len(deg)):

        deg_i = deg[i]
        node_i = data.G.nodes()[i]
        node_i_idx = data.id2idx[node_i]
        feature[node_i_idx, int(deg_i/(binn+ 1))] = 1
    return feature


if __name__ == "__main__":
    args = parse_args()
    data1 = Dataset(args.input_data1)
    data2 = Dataset(args.input_data2)

    feature1, feature2 = create_features(data1, data2, args.feature_dim)
    np.save(args.input_data1 + 'feats.npy', feature1)
    np.save(args.input_data2 + 'feats.npy', feature2)