from input.semi_synthetic import SemiSynthetic
from input.fully_synthetic import FullySynthetic
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Erdos Renyi Graph Generation")
    parser.add_argument('--output_path', default="data/fully-synthetic")
    parser.add_argument('--n', default=10000, type=int)
    parser.add_argument('--p', default=0.5, type=float)
    return parser.parse_args()


def gen_fully(full, n_nodes,p):
    name_p = str(p).replace("0.","")
    outdir = full+"/erdos-renyi-n{}-p{}".format(n_nodes,name_p)
    FullySynthetic.generate_erdos_renyi_graph(outdir,n_nodes=n_nodes,p_edge_creation=p)
    return outdir


if __name__ == "__main__":
    args = parse_args()
    out_dir = gen_fully(args.output_path, args.n, args.p)
