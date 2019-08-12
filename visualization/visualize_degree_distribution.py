from __future__ import print_function, division
import json
import sys
import argparse
from shutil import copyfile

import networkx as nx
from networkx.readwrite import json_graph
import pdb

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import time



def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised graphSAGE")
    parser.add_argument('--input_path',  default="$HOME/dataspace/graph/ppi/sub_graph/subgraph0/graphsage")
    parser.add_argument('--prefix', default='ppi')
    parser.add_argument('--name', default='')
    return parser.parse_args()



def visualize_degree_distribution(deg, out_path, name):
    deg = np.sort(deg)
    unique, counts = np.unique(deg, return_counts=True)
    y_max = max(counts)
    std = np.std(deg)
    mean = np.mean(deg)
    print("Standard deviation: ", std)
    print("Mean degree: ", mean)
    print("Max degree: ", max(deg))
    quality = round(max(deg) / std, 1)
    print("Quality: ", quality)
    plt.hist(np.array(deg), max(deg))
    plt.vlines(mean - std, 0, y_max, linestyle='dashed', linewidth=0.8, label='std line')
    plt.vlines(mean + std, 0, y_max, linestyle='dashed', linewidth=0.8)
    plt.vlines(mean, 0, y_max, color='red', linestyle='dashed', linewidth=1, label='mean line')
    plt.text(mean +std*0.05, y_max*2/3, 'mean = ' + str(round(mean, 1)) + ', std = ' + str(round(std, 1)), color='b')
    plt.text(max(deg) / 2, y_max/2, 'quality = ' + str(quality))
    plt.xlabel('degree')
    plt.ylabel('num nodes')
    plt.title('deg')
    plt.grid(True)
    plt.savefig(out_path + name + "_deg.png")

def main():
    args = parse_args()
    print(args)
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    G_data = json.load(open(args.input_path + '/' + args.prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    print(nx.info(G))
    deg = np.zeros((len(G.nodes()),)).astype(int)
    for i in range(len(deg)):
        deg[i] = len(G.neighbors(G.nodes()[i]))

    visualize_degree_distribution(deg, args.input_path, args.name)
    print("DONE!!!")

if __name__ == "__main__":
    main()