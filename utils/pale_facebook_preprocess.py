from collections import defaultdict
import numpy as np
import json
import os
import networkx as nx
from networkx.readwrite import json_graph
from edgelist_to_graphsage import edgelist_to_graphsage

def filter_file(dataset, dir):
    lines = []
    with open(dir + "/" + dataset) as fp:
        for line in fp:
            entities = line.split()
            lines.append("{0}\t{1}".format(entities[0], entities[1]))
    
    if not os.path.exists(dir + "/edgelist/"):
        os.makedirs(dir + "/edgelist/")
    
    outfile = dir + "/edgelist/" + dataset + ".edgelist"
    
    with open(outfile, "w") as fp:        
        fp.write("\n".join([line for line in lines]))
        print("Edgelist has been stored in: {0}".format(outfile))

def filter_nodes_by_degree(dataset, dir, threshold_degree):
    edgelist_file = dir + "/edgelist/" + dataset + ".edgelist"
    G = nx.read_edgelist(edgelist_file)
    print("Filter graph with threshold degree = {0}".format(threshold_degree))
    print("Original graph info")
    print(nx.info(G))
    
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
    min_deg = 10000
    for node in G.nodes():
        min_deg = min(G.degree(node), min_deg)
    print("Min degree = {0}".format(min_deg))    
    print("Edgelist has been stored in: {0}".format(edgelist_file))
    nx.write_edgelist(G, path = edgelist_file , delimiter="\t", data=['weight'])
            
if __name__ == "__main__":
    datadir = "/home/trunght/workspace/network_embeddings/GraphSAGE_pytorch/example_data/pale_facebook"
    dataset = "pale_facebook"
    filter_file(dataset, datadir)     
    filter_nodes_by_degree(dataset, datadir, 5)
    edgelist_to_graphsage(dataset, datadir)
