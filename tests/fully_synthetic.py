from input.semi_synthetic import SemiSynthetic
from input.fully_synthetic import FullySynthetic
# outdir='../dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/'
# FullySynthetic.generate_erdos_renyi_graph(outdir,n_nodes=10000,p_edge_creation=0.1)
# semiSynthetic = SemiSynthetic(outdir+'/graphsage',outdir+'/random-a1-d1/')
# semiSynthetic.generate_random_clone_synthetic(0.1,0.1)

full="../dataspace/graph/fully-synthetic/"

def gen_fully(n_nodes,p):
    name_p = str(p).replace("0.","")
    outdir = full+"/erdos-renyi-n{}-p{}".format(n_nodes,name_p)
    FullySynthetic.generate_erdos_renyi_graph(outdir,n_nodes=n_nodes,p_edge_creation=p)
    return outdir

def gen_randomclone(origin_dir, a,d):
    semiSynthetic = SemiSynthetic(
        origin_dir+"/graphsage",
        origin_dir+"/random-a{}-d{}".format(str(a).replace("0.",""), str(d).replace("0.","")))
    semiSynthetic.generate_random_clone_synthetic(a,d)

out = {}
out["10000-05"]=gen_fully(10000,.05)
gen_randomclone(out["10000-05"],.1,.1)

out["10000-1"]=gen_fully(10000,.1)
gen_randomclone(out["10000-1"],.1,.1)

out["10000-2"]=gen_fully(10000,.2)
gen_randomclone(out["10000-2"],.1,.1)

out["10000-3"]=gen_fully(10000,.3)
gen_randomclone(out["10000-3"],.1,.1)

out["1000-2"]=gen_fully(1000,.2)
gen_randomclone(out["1000-2"],.1,.1)

out["100000-2"]=gen_fully(100000,.2)
gen_randomclone(out["100000-2"],.1,.1)

out["1000000-2"]=gen_fully(1000000,.2)
gen_randomclone(out["1000000-2"],.1,.1)

out["10000000-2"]=gen_fully(10000000,.2)
gen_randomclone(out["10000000-2"],.1,.1)