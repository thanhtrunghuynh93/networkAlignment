from input.semi_synthetic import SemiSynthetic
from input.fully_synthetic import FullySynthetic
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Erdos Renyi Graph Generation")
    parser.add_argument('--output_path', default="data/fully-synthetic")
    parser.add_argument('--n', default=10000, type=int, help="Number of nodes")
    parser.add_argument('--aver', default=5, type=float, help="Average degree")
    return parser.parse_args()


def gen_fully(full, n_nodes,n_edges, p):
    name_p = str(int(p))
    outdir = full+"/erdos-renyi-n{}-p{}".format(n_nodes,name_p)
    FullySynthetic.generate_erdos_renyi_graph(outdir,n_nodes=n_nodes,n_edges=n_edges)
    return outdir


if __name__ == "__main__":
    args = parse_args()
    num_edges = args.aver * args.n / 2
    out_dir = gen_fully(args.output_path, args.n, num_edges, args.aver)
