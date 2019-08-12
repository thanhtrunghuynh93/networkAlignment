import numpy as np
import os
import argparse
import networkx as nx
from networkx.readwrite import json_graph
import json
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Merge two graphs into one.")
    parser.add_argument('--prefix1', default="example_data/ppi/graphsage/ppi")
    parser.add_argument('--prefix2', default="example_data/ppi/graphsage/ppi")
    parser.add_argument('--out_dir', default="example_data/ppi/merge/")
    parser.add_argument('--out_prefix', default="ppi")
    parser.add_argument('--groundtruth', default="example_data/ppi/permut/dictionaries/groundtruth")
    return parser.parse_args()

def load_graph(prefix):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    res = json_graph.node_link_data(G)

    id_map = json.load(open(prefix + "-id_map.json"))

    features = None
    if os.path.isfile(prefix + "-feats.npy"):
        features = np.load(prefix+"-feats.npy")
    return G, id_map, res, features

def load_groundtruth(groundtruth_file):
    gt = {}
    with open(groundtruth_file, 'r') as file:
        for line in file:
            src, trg = line.split()
            gt[src] = trg
    return gt

def generate_graphs(prefix1, prefix2, groundtruth_file):
    G1, id_map1, res1, features1 = load_graph(prefix1)
    G2, id_map2, res2, features2 = load_graph(prefix2)
    groundtruth = load_groundtruth(groundtruth_file)

    new_nodes_ids1 = np.arange(len(G1.nodes()))
    new_nodes_ids2 = np.arange(len(G1.nodes()), len(G1.nodes())+len(G2.nodes()))
    new_nodes_ids1 = list(map(str, new_nodes_ids1)) # source nodes ids
    new_nodes_ids2 = list(map(str, new_nodes_ids2)) # target nodes ids

    new_id_map = {}
    for node_id in new_nodes_ids1:
        new_id_map[node_id] = int(node_id)
    for node_id in new_nodes_ids2:
        new_id_map[node_id] = int(node_id)

    new_gt = {}
    for src, trg in groundtruth.items():
        new_src_id = new_nodes_ids1[id_map1[src]]
        new_trg_id = new_nodes_ids2[id_map2[trg]]
        new_gt[new_src_id] = new_trg_id

    new_nodes = []
    for idx, node in enumerate(res1["nodes"]):
        node["id"] = new_nodes_ids1[idx]
        new_nodes.append(node)
    for idx, node in enumerate(res2["nodes"]):
        node["id"] = new_nodes_ids2[idx]
        new_nodes.append(node)


    new_links = []
    for link in res1["links"]:
        new_links.append({
            'source': new_id_map[new_nodes_ids1[link["source"]]],
            'target': new_id_map[new_nodes_ids1[link["target"]]]
        })
    for link in res2["links"]:
        new_links.append({
            'source': new_id_map[new_nodes_ids2[link["source"]]],
            'target': new_id_map[new_nodes_ids2[link["target"]]]
        })

    new_features = None
    if features1 is not None and features2 is not None:
        if features1.shape[1] != features2.shape[1]:
            print("Can not create new features due to different features shape.")
        new_features = np.zeros((features1.shape[0]+features2.shape[0], features1.shape[1]))
        for i, feat in enumerate(features1):
            new_features[new_id_map[new_nodes_ids1[i]]] = feat
        for i, feat in enumerate(features2):
            new_features[new_id_map[new_nodes_ids2[i]]] = feat

    new_res = res1
    new_res["nodes"] = new_nodes
    new_res["links"] = new_links
    return new_res, new_id_map, new_features, new_nodes_ids1, new_nodes_ids2, new_gt

def save_graph(args, res, id_map, features, source_ids, target_ids, gt):
    out_dir, out_prefix = args.out_dir, args.out_prefix
    if not os.path.exists(out_dir+"/graphsage"):
        os.makedirs(out_dir+"/graphsage/")
    if not os.path.exists(out_dir+"/edgelist"):
        os.makedirs(out_dir+"/edgelist")
    if not os.path.exists(out_dir+"/dictionaries"):
        os.makedirs(out_dir+"/dictionaries")

    with open(out_dir+"/graphsage/"+out_prefix+"-G.json", "w") as file:
        file.write(json.dumps(res))
    with open(out_dir+"/graphsage/"+out_prefix+"-id_map.json", "w") as file:
        file.write(json.dumps(id_map))
    if features is not None:
        np.save(out_dir+"/graphsage/"+out_prefix+"-feats.npy", features)
    np.save(out_dir+"/graphsage/"+out_prefix+"-source_ids.npy", source_ids)
    np.save(out_dir+"/graphsage/"+out_prefix+"-target_ids.npy", target_ids)

    nx.write_edgelist(json_graph.node_link_graph(res),
        path=out_dir + "/edgelist/" + out_prefix + ".edgelist" , delimiter=" ", data=['weight'])

    with open(args.out_dir + "/dictionaries/groundtruth", 'w') as f:
        for src, trg in gt.items():
            f.write("{} {}\n".format(src, trg))
        f.close()

    print("Graph has been saved to {}".format(out_dir))

def test_graph(args):
    G, idmap,_,_ = load_graph(args.out_dir+"/graphsage/"+args.out_prefix)
    nodes = G.nodes()
    for idx, node in enumerate(nodes):
        assert idmap[node] == idx, "Test fail"
    print("Test passed.")
    # nodes = list(map(int, nodes))
    # nodes_expected = np.arange(len(nodes)).tolist()
    # print("Test graph id map:", nodes == nodes_expected)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    res, id_map, features, source_ids, target_ids, gt = generate_graphs(args.prefix1, args.prefix2, args.groundtruth)
    save_graph(args, res, id_map, features, source_ids, target_ids, gt)
    test_graph(args)
