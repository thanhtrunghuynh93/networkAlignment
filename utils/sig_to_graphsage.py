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

class SigLinkagePreprocessor:
    def __init__(self, data_dir, dataset_name1, dataset_name2, output_dir):
        self.data_dir = data_dir
        self.dataset_name1 = dataset_name1
        self.dataset_name2 = dataset_name2
        self.output_dir = output_dir

    def read_edge_list_file(dataset_name):
        file = open(data_dir + dataset_name + '_sub_network.txt')
        lines = file.readlines()
        for line in lines:
            token = line.split('\t')
            

    def process(self):
        node_list1, edge_list1 = self.read_edge_list_file(self.dataset_name1)
        node_list2, edge_list2 = self.read_edge_list_file(self.dataset_name2)