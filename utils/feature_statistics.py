from __future__ import print_function, division
import numpy as np

import pdb
import argparse
import os

def feature_statistic(prefix):
           
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
        frequencies = np.zeros(feats.shape[1])
        for feat in feats:
            frequencies += feat
        print("Feature frequencies:")
        print(frequencies)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Randomly remove edges and generate dict.")
    parser.add_argument('--prefix', default=None, help='Path to groundtruth file')
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    feature_statistic(args.prefix)
