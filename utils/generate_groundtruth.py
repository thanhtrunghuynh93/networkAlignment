import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate grountruth self dictionary for vecmap")
    parser.add_argument('--nodes_file', default="all.txt",  help="Nodes file")
    parser.add_argument('--out_dir', default="", help="Output file")
    return parser.parse_args()

def load_nodes_file(nodes_file):
    nodes = []
    if os.path.exists(nodes_file):
        with open(nodes_file) as fp:
            for line in fp:
                nodes.append(line.strip())                    
    return nodes

def convert_and_save(nodes, out_dir):
    """
    nodes: list of nodes' ids
    out_file: directory to save output files.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outfile = open(os.path.join(out_dir, 'groundtruth.dict'), 'w+')
    for node in nodes:
        outfile.write("{0} {1}".format(node, node))
        outfile.write('\n')
    outfile.close()
    
if __name__ == '__main__':
    args = parse_args()
    nodes = load_nodes_file(args.nodes_file)
    convert_and_save(nodes, args.out_dir)