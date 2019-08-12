# from input.semi_synthetic import SemiSynthetic
# from input.fully_synthetic import FullySynthetic
#
# networkx_dir = '../dataspace/graph/ppi/subgraphs/subgraph3/graphsage/'
# #
# #
# output_dir1 = '../dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s09-c09-1/'
# output_dir2 = '../dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s09-c09-2/'
# semiSynthetic = SemiSynthetic(
#     networkx_dir,
#     output_dir1,
#     output_dir2=output_dir2)
# semiSynthetic.generate_PALE_synthetic()
#
# output_dir1 = '../dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s05-c09-1/'
# output_dir2 = '../dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s05-c09-2/'
# semiSynthetic = SemiSynthetic(networkx_dir,
#                               output_dir1,
#                               output_dir2)
# semiSynthetic.generate_PALE_synthetic()
#
# output_dir1 = '../dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s09-c06-1/'
# output_dir2 = '../dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/PALE-s09-c06-2/'
# semiSynthetic = SemiSynthetic(networkx_dir,output_dir1,output_dir2)
# semiSynthetic.generate_PALE_synthetic()


from input.semi_synthetic import SemiSynthetic
from input.fully_synthetic import FullySynthetic

def gen_REGAL(d, ppi, p_change_feats=None, seed=1):
    name_d = str(d).replace("0.","")
    networkx_dir = ppi+'/graphsage'
    if p_change_feats is not None:
        outdir = ppi+'/semi-synthetic/REGAL-d{}-pfeats{}-seed{}/'.format(name_d,
                                                                  str(p_change_feats).replace("0.",""), seed)
    else:
        outdir = ppi + '/REGAL-d{}-seed{}/'.format(name_d, seed)
    semiSynthetic = SemiSynthetic(networkx_dir, outdir, seed = seed)
    semiSynthetic.generate_random_clone_synthetic(0, d, p_change_feats=p_change_feats)

def gen_PALE(s,c, seed=1, ppi=None):
    name_s = str(s).replace("0.","")
    name_c = str(c).replace("0.","")
    networkx_dir = ppi+'/graphsage'
    outdir = ppi+'/semi-synthetic/PALE-s{}-c{}-seed{}'.format(name_s, name_c, seed)
    semiSynthetic = SemiSynthetic(networkx_dir, outdir+"-1", outdir+"-2", seed=seed)
    semiSynthetic.generate_PALE_synthetic(s,c)


# ppi="../dataspace/graph/fb-tw-data/twitter/"

# for pdel_edges in [.001,.005,.01,.05,.1,.2]:
#     # for p_change_feats in [.05, .1, .2]:
#     gen_REGAL(pdel_edges)

def dell(name):
    ppi="data/" + name
    # for p_feats in [.1, .2, .3, .4, .5]:
    #    gen_REGAL(0.05, ppi, p_feats)
    # # for seed in range(1, 21):
    for pdel_edges in [.01, .05, .1, .2]:
        gen_REGAL(pdel_edges, ppi)
    # for seed in range(1, 21):
    #     for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
    #         gen_PALE(s, 0.9, seed, ppi)
    #     for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
    #         gen_PALE(0.6, c, seed, ppi)
            

if __name__ == "__main__":
    dell("econ")
    dell("ppi")
    dell("bn")

