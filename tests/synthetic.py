from input.semi_synthetic import SemiSynthetic
from input.fully_synthetic import FullySynthetic
import numpy as np

n_nodes = [1000, 2000, 5000, 10000, 20000]
ks = [20,60,100,200,350]

for n in n_nodes:
    for k in ks:
        #np.random.seed(123)
        outdir="../dataspace/graph/fully-synthetic/small-world-n{}-k{}-p5-seed123/".format(n, k)
        FullySynthetic.generate_small_world_graph(outdir,n_nodes=n,k_neighbors=k, p_edge_modify=0.5, feature_dim=16)
        semiSynthetic = SemiSynthetic(outdir+'/graphsage',outdir+'/random-d01')
        semiSynthetic.generate_random_clone_synthetic(0, 0.01)
