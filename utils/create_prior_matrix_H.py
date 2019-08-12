import argparse
from input.dataset import Dataset
import numpy as np
import os
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Create prior H matrix used for IsoRank and FINAL.")
    parser.add_argument('--source_dataset', default='../dataspace/graph/douban/online/graphsage',
                        help="Input directory for source dataset.")
    parser.add_argument('--target_dataset', default='../dataspace/graph/douban/offline/graphsage',
                        help="Input directory for target dataset.")
    parser.add_argument('--out_dir', default='../dataspace/douban/',
                        help="Output directory for saving H.npy.")
    return parser.parse_args()

def create_H(args):
    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    src_degrees = np.array(source_dataset.get_nodes_degrees())
    trg_degrees = np.array(target_dataset.get_nodes_degrees())
    # import pdb
    # pdb.set_trace()

    # src_degrees = src_degrees/src_degrees.max()
    # trg_degrees = trg_degrees/trg_degrees.max()
    #
    # distance_matrix = np.zeros((len(src_degrees), len(trg_degrees)))
    # for src_idx, src_deg in enumerate(src_degrees):
    #     for trg_idx, trg_deg in enumerate(trg_degrees):
    #         distance_matrix[src_idx,trg_idx] = np.abs(src_deg-trg_deg)
    # max_distance = distance_matrix.max()
    # H = 1-distance_matrix/max_distance
    # H = H.T

    H = np.zeros((len(trg_degrees),len(src_degrees)))
    for i in range(H.shape[0]):
        H[i,:] = np.abs(trg_degrees[i]-src_degrees)/max([src_degrees.max(),trg_degrees[i]])
    H = H/H.sum()

    # H = np.zeros((len(trg_degrees),len(src_degrees)))
    # for i, trg_deg in enumerate(trg_degrees):
    #     for j, src_deg in enumerate(src_degrees):
    #         H[i,j]=1-min([src_deg,trg_deg])/max([src_deg,trg_deg])
    # idxs_trg = np.random.choice(H.shape[0],2000000,replace=True)
    # idxs_src = np.random.choice(H.shape[1],2000000,replace=True)
    # H[idxs_trg,idxs_src]=0

    print("H shape: ", H.shape)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    np.save(args.out_dir+"/H2.npy",H)
    print("H has been saved to ", args.out_dir)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    create_H(args)