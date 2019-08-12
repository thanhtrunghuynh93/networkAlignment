import scipy.io
import scipy.sparse
import argparse
import numpy as np
import os
import json
import re
import pandas
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Convert data from graphsage format to .mat format.")
    parser.add_argument('--prefix', default="example_data/douban/online/graphsage/online", help="Dataset prefix.")
    parser.add_argument('--prefix2', default="example_data/douban/offline/graphsage/offline", help="Dataset prefix.")
    parser.add_argument('--groundtruth', default="example_data/douban/dictionaries/groundtruth.dict", help="Ground truth file.")
    parser.add_argument('--out', default="example_data/douban/matlab/douban", help="Output prefix.")
    return parser.parse_args()

def convert(args):
    G = json.load(open(args.prefix+"-G.json", "r"))
    id_map = json.load(open(args.prefix+"-id_map.json", "r"))
    # n1, offline, offline_node_label,
    groundtruth = pandas.read_csv(args.groundtruth, sep=" ").values.astype(np.uint16)

    n1 = np.array([[len(G["nodes"])]], dtype=np.uint16)

    # build links
    data = np.array([1.0, 1.0]*len(G["links"]))
    row = []
    col = []
    for link in G["links"]:
        row.append(id_map[str(link["source"])])
        col.append(id_map[str(link["target"])])
        row.append(id_map[str(link["target"])])
        col.append(id_map[str(link["source"])])
    links1 = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(G["nodes"]), len(G["nodes"])))
    # end build links

    # build offline_node_label
    data = []
    row = []
    col = []
    for node in G["nodes"]:
        mask = np.array(node["feature"]) > 0.0
        indexes = mask.nonzero()[0]
        row += [id_map[str(node["id"])]]*len(indexes)
        col += indexes.tolist()
        data += np.array(node["feature"])[indexes].tolist()
    node_label1 = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(G["nodes"]), len(G["nodes"][0]["feature"])))





    G = json.load(open(args.prefix2+"-G.json", "r"))

    n2 = np.array([[len(G["nodes"])]], dtype=np.uint16)

    # build links
    data = np.array([1.0, 1.0]*len(G["links"]))
    row = []
    col = []
    for link in G["links"]:
        row.append(id_map[str(link["source"])])
        col.append(id_map[str(link["target"])])
        row.append(id_map[str(link["target"])])
        col.append(id_map[str(link["source"])])
    links2 = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(G["nodes"]), len(G["nodes"])))
    # end build links

    # build offline_node_label
    data = []
    row = []
    col = []
    for node in G["nodes"]:
        mask = np.array(node["feature"]) > 0.0
        indexes = mask.nonzero()[0]
        row += [id_map[str(node["id"])]]*len(indexes)
        col += indexes.tolist()
        data += np.array(node["feature"])[indexes].tolist()
    node_label2 = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(G["nodes"]), len(G["nodes"][0]["feature"])))


    #
    #
    # TODO: Check code here
    # H = np.random.uniform(0, 1, size=(n2[0][0], n1[0][0]))
    # H = H / H.sum()
    # H1 = H.T

    H = np.zeros((n2[0][0], n1[0][0]))
    H1 = H.T
    # end TODO
    #
    #

    # build mat
    mat = {
        "n1": n1,
        "online": links1,
        "online_node_label": node_label1,
        "n2": n2,
        "offline": links2,
        "offline_node_label": node_label2,
        "ground_truth": groundtruth,
        "H": H,
        "H1": H1
    }

    # mat1 = {
    #     "n1": n1,
    #     "edge": links1,
    #     "node_label": node_label1,
    #     "H": H,
    #     "groundtruth": groundtruth
    # }
    # mat2 = {
    #     "n2": n2,
    #     "edge": links2,
    #     "node_label": node_label2,
    #     "H1": H1
    # }
    output_dir = "/".join(re.split(r"[\\\/]", args.out)[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scipy.io.savemat(args.out+".mat", mat, do_compression=True)
    # scipy.io.savemat(args.out+"1.mat", mat1)
    # scipy.io.savemat(args.out+"2.mat", mat2)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    convert(args)
