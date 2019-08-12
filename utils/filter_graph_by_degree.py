from __future__ import print_function, division
from pathlib import Path
import numpy as np
import random
import json
import sys
import os
import argparse
from shutil import copyfile

import networkx as nx
from networkx.readwrite import json_graph
import pdb

def filter_graph_by_degree(G, threshold_degree = 5):
    #Initialize num_node_remove    
    num_node_remove = 10000
    total_node_remove = 0
    num_round = 0
    while num_node_remove > 0:
        print("Round {0}".format(num_round))
        num_node_remove = 0
        for node in G.nodes():
            if(G.degree(node) < threshold_degree):
                num_node_remove += 1
                G.remove_node(node)
        total_node_remove += num_node_remove
        num_round += 1
        print("Filtered {0} nodes".format(num_node_remove))
    
    print("Filtered total {0} nodes".format(total_node_remove))
    print("Graph info after filtering")
    print(nx.info(G))    