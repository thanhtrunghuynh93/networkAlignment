import numpy as np
import os
import argparse
import json
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Split embedding into source and target embedding.")
    parser.add_argument('--embedding_file', default="visualize/ppi/embedding.emb")
    parser.add_argument('--source_ids', default="example_data/ppi/merge/graphsage/ppi-source_ids.npy",
        help="File that contains source nodes' id in merged graph.")
    parser.add_argument('--target_ids', default="example_data/ppi/merge/graphsage/ppi-target_ids.npy",
        help="File that contains target nodes' id in merged gragh.")
    parser.add_argument('--out_dir', default="visualize/ppi/")
    return parser.parse_args()

def split_embeds(args):
    embed_file = args.embedding_file
    source_ids, target_ids, out_dir = np.load(args.source_ids), np.load(args.target_ids), args.out_dir
    embeds = {}
    source_embeds = {}
    target_embeds = {}

    with open(embed_file, "r") as file:
        header = file.readline().split(' ')
        count = int(header[0])
        dim = int(header[1])

        for i in range(count):
            node_id, vec = file.readline().split(' ', 1)
            embeds[node_id] = np.fromstring(vec, sep=' ', dtype=np.float64)

    for id in source_ids:
        source_embeds[id] = embeds[id]
    for id in target_ids:
        target_embeds[id] = embeds[id]

    with open(out_dir+"/source.emb", "w") as file:
        file.write("%s %s\n"%(len(source_ids), dim))
        for id, embed in source_embeds.items():
            txt_vector = ["%s" % embed[j] for j in range(dim)]
            file.write("%s %s\n"%(id, " ".join(txt_vector)))

    with open(out_dir+"/target.emb", "w") as file:
        file.write("%s %s\n"%(len(target_ids), dim))
        for id, embed in target_embeds.items():
            txt_vector = ["%s" % embed[j] for j in range(dim)]
            file.write("%s %s\n"%(id, " ".join(txt_vector)))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    split_embeds(args)
