import subprocess
from input.semi_synthetic import SemiSynthetic
from utils.connected_components import *
from input.fully_synthetic import FullySynthetic

HOME='..'
FULLY=HOME+'/dataspace/graph/fully-synthetic/connected-components/'

def generate_graph(nc, seed):
    out_dir = FULLY+'small-world-n1000-k10-p5-nc'+str(nc)+'-seed'+str(seed)
    graph = create_graph(n=1000, nc=nc, feature_dim=16, k=10, p=0.5, seed=seed)
    features = get_features_from_graph(graph)
    FullySynthetic.save_graph(out_dir, graph, features)
    return out_dir

# def shuffle_graph(nc, seed):
#     dirr = FULLY+"smallworld-n1000-k10-p5-nc"+str(nc)+'-seed'+str(seed) +'/random-d01'
#     command = [
#         'python',
#         'utils/shuffle_graph.py',
#         '--input_dir',
#         dirr,
#         '--out_dir',
#         dirr+'--1'
#     ]
#     p = subprocess.Popen(command)
#     p.communicate()
#     command = [
#         'rm',
#         '-r',
#         dirr
#     ]
#     p = subprocess.Popen(command)
#     p.communicate()
#     command = ['mv', dirr+'--1', dirr]
#     p = subprocess.Popen(command)
#     p.communicate()

for seed in range(100, 150):
    for nc in [1,2,3,4,5,6]:
        out_dir = generate_graph(nc, seed)
        networkx_dir = out_dir+"/graphsage"
        semiSynthetic = SemiSynthetic(networkx_dir, out_dir+"/random-d01")
        semiSynthetic.generate_random_clone_synthetic(0,0.01)
