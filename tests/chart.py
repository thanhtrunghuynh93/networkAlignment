import matplotlib.pyplot as plt
import re
import pdb
import numpy as np

import matplotlib
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams.update(pgf_with_rc_fonts)

def get_info(text):
    accs = re.findall(r"Accuracy: .+?[\"\n]", text)
    maps = re.findall(r"MAP: .+?[\"\n]", text)
    aucs = re.findall(r"AUC: .+?[\"\n]", text)
    hits = re.findall(r"Hit-precision: .+?[\"\n]", text)
    top5s = re.findall(r"Precision_5: .+?[\"\n]", text)
    top10s = re.findall(r"Precision_10: .+?[\"\n]", text)
    times = re.findall(r"Time: .+?(?=[\"\n\t])", text)

    accs = list(map(lambda x: x.replace("Accuracy: ", "").strip(), accs))
    maps = list(map(lambda x: x.replace("MAP: ", "").strip(), maps))
    aucs = list(map(lambda x: x.replace("AUC: ", "").strip(), aucs))
    hits = list(map(lambda x: x.replace("Hit-precision: ", "").strip(), hits))
    top5s = list(map(lambda x: x.replace("Precision_5: ", "").strip(), top5s))
    top10s = list(map(lambda x: x.replace("Precision_10: ", "").strip(), top10s))
    times = list(map(lambda x: x.replace("Time: ", "").replace("Time used for alignment: ","").strip(), times))

    def to_float(x): return np.array(list(map(float, x)))
    accs = to_float(accs)
    maps = to_float(maps)
    aucs=to_float(aucs)
    hits=to_float(hits)
    top5s=to_float(top5s)
    top10s=to_float(top10s)
    times=to_float(times)

    return (accs,
        maps,
        aucs,
        hits,
        top5s,
        top10s,
        times)
    # file = open(text, 'r')
    # results = []
    # for line in file:
    #     result_line = [float(ele) for ele in line.split()]
    #     results.append(result_line)
    # results = np.array(results)
    # file.close()
    return results


def line_chart(data, xpoints, xtitle, ytitle, filename, models, yticks=None, add_Legend=False):
    """

    :param data: np array shape (n_models, n_xpoints)
    :param xpoints:
    :param xtitle:
    :param ytitle:
    :return:
    """

    styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']
    # colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)
    for i in range(len(models)):
        ax.plot(np.arange(len(xpoints)), data[i], styles[i], label=models[i], markersize=20, fillstyle=None) #, color=colors[i])
    # plot(x,h1, , marker="^",ls='--',label='GNE',fillstyle='none')

    plt.xticks(np.arange(len(xpoints)), xpoints)
    plt.xlabel(xtitle, fontsize=30)
    # plt.ylabel(ytitle)
    plt.title(ytitle, fontsize=30)

    if add_Legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, bbox_to_anchor=(1,1.1))
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        print(filename)
        fig.savefig(filename, bbox_inches='tight')



def barchart(data, cluster_names, xtitle, ytitle, filename, yticks=None, add_Legend = False):
    """
    :param cluster_names: list of cluster names
    :param data: np array of shape (n, len(cluster_names)) where n is the number of bars in one cluster
    :param xtitle: title of x axis
    :param ytitle: title of y axis
    :return:
    """
    n_bars_in_cluster = data.shape[0]
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    index = np.arange(len(cluster_names))
    bar_width = 0.1
    opacity = 0.8

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.9])

    _models = models

    colors = ['#003300','#228B22',  '#66ff66', '#ffffff']
    # print(n_bars_in_cluster)
    # raise Exception

    for i in range(n_bars_in_cluster):
        ax.bar(index + bar_width * i, data[i], bar_width, alpha=opacity, label=_models[i], color=colors[i], edgecolor='#000000')

    plt.xlabel(xtitle, fontsize=30)
    # plt.ylabel(ytitle)
    plt.title(ytitle, fontsize=30)
    plt.xticks(index + bar_width * 3, cluster_names)
    if yticks is not None:
        plt.yticks(*yticks)

    if add_Legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, bbox_to_anchor=(1,1.1))
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        fig.savefig(filename, bbox_inches='tight')

# def line_chart(data, xpoints, xtitle, ytitle, filename, add_FAN=False, yticks=None, add_Legend = False):
#     """

#     :param data: np array shape (n_models, n_xpoints)
#     :param xpoints:
#     :param xtitle:
#     :param ytitle:
#     :return:
#     """
#     styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-']
#     # styles = ['^--']*7

#     fig = plt.figure(dpi=100)
#     ax = fig.add_subplot(111)
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width, box.height*0.9])

#     if add_FAN:
#         _models = models_extra
#     else:
#         _models = models

#     # _models = ["REGAL","FINAL","BigAlign"]
#     # data = data[[0,1,3],:]
#     for i in range(len(_models)):
#         ax.plot(np.arange(len(xpoints)), data[i], styles[i], label=_models[i], markersize=12)

#     plt.xticks(np.arange(len(xpoints)), xpoints)
#     if yticks is not None:
#         plt.yticks(*yticks)
#     plt.xlabel(xtitle)
#     plt.ylabel(ytitle)

#     if add_Legend:
#         handles, labels = ax.get_legend_handles_labels()
#         lgd = ax.legend(handles, labels, bbox_to_anchor=(1,1.1))
#         fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
#     else:
#         fig.savefig(filename, bbox_inches='tight')
#     # fig.savefig(filename,  bbox_inches='tight')
#     # plt.show()

def draw_realdata(add_FAN=False):
    print("realdata")
    cluster_names = ["douban", 'flickr-lastfm','flickr-myspace','fb-tw', 'fq-tw']
    text = open("tests/raw-data/real.txt").read()
    data = get_info(text)
    accs = data[0].reshape(len(models), -1)
    barchart(accs, cluster_names, "Dataset", "Accuracy",
             filename="visualization/images/realdata.png", add_FAN=add_FAN)

def draw_semi_regal(add_FAN=False):
    print("semi_regal")
    xpoints = ["0", "0.001", "0.005", "0.01", "0.05", "0.1", "0.2"]
    text = open("tests/raw-data/semi_regal.txt").read()
    data = get_info(text)
    accs = data[0].reshape(len(models), len(xpoints))
    xpoints = ["0", "0.01", "0.05", "0.1", "0.2"]
    accs = accs[:, [0,3,4,5,6]]
    line_chart(accs, xpoints, "Edges removal ratio", "Accuracy",
               filename="visualization/images/semi_regal_test.png", add_FAN=add_FAN)

def draw_delnodes(add_FAN=False):
    print("delnodes")
    # xpoints = ['0.1', '0.2', '0.3', '0.4', '0.5']
    xpoints = ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
    text = open("tests/raw-data/delnodes.txt").read()
    data = get_info(text)
    accs = data[0].reshape(len(models), len(xpoints))
    # accs = accs[:, [0,3,4,5,6,7]]
    # xpoints = ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
    line_chart(accs, xpoints, "Nodes removal ratio", "Accuracy",
               filename="visualization/images/delnodes_ppi.png", add_FAN=add_FAN)

def draw_density(extra=None, add_FAN=False, models=None):
    xpoints = ["4", "60", "100", "200", "700", "999"]
    text = open("tests/raw-data/density.txt").read()

    data = get_info(text)
    if add_FAN:
        accs = data[0].reshape(len(models_extra), len(xpoints))
    else:
        accs = data[0].reshape(len(models), len(xpoints))

    if extra is not None:
        filename = "visualization/images/density-{}.png".format(extra)
    else:
        filename = "visualization/images/density.png"
    line_chart(accs, xpoints, "k", "Accuracy", filename=filename, models=models)

def draw_connectivity(extra=None, add_FAN=False):
    print("connectivity-{}".format(extra))
    xpoints = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
    if extra is not None:
        text = open("tests/raw-data/connectivity-{}.txt".format(extra)).read()
    else:
        text = open("tests/raw-data/connectivity.txt").read()

    data = get_info(text)
    if add_FAN:
        accs = data[0].reshape(len(models_extra), len(xpoints))
    else:
        accs = data[0].reshape(len(models), len(xpoints))

    if extra is not None:
        filename = "visualization/images/connectivity-{}.png".format(extra)
    else:
        filename = "visualization/images/connectivity.png"
    line_chart(accs, xpoints, "p", "Accuracy", filename=filename, add_FAN=add_FAN)

def draw_changefeats():
    print("changefeats")
    xpoints = [0, .05, .1, .2, .3, .4, .5]
    text = open("tests/raw-data/changefeats-d01.txt").read()

    data = get_info(text)
    accs = data[0].reshape(len(models), len(xpoints))

    xpoints = [0, .1, .2, .3, .4, .5]
    accs = accs[:,[0, 2,3,4,5,6]]

    filename = "visualization/images/changefeats-d01-econ.png"
    line_chart(accs, xpoints, "Features changing ratio", "Accuracy", filename=filename, add_Legend=True)

def draw_time():
    print("time")
    xpoints = [500, 1000, 2000, 5000, 10000, 20000]
    text = open("tests/raw-data/time.txt").read()

    data = get_info(text)
    accs = data[-1].reshape(len(models), len(xpoints))
    accs = np.log10(accs+1)
    xpoints = ["0.5", "1", "2", "5", "10", "20"]
    accs = accs[:, 1:]
    xpoints = ["1k", "2k", "5k", "10k", "20k"]
    filename = "visualization/images/time.png"
    barchart(accs, xpoints, "Number of nodes", "Time(log(s))", filename,
        yticks=([0,2,4], ["0", "2", "4"]))

    # line_chart(accs, xpoints, "Number of nodes", "Time (s)", filename=filename,
    #     yticks=(np.arange(accs.min(), accs.max(), 5000), ["0", "5k", "10k"]))

def draw_timek():
    print("time")
    xpoints = "2 3 4 10 20 40 60 80 100 200 350 500 700 900 999".split()
    text = open("tests/raw-data/timek.txt").read()

    data = get_info(text)
    accs = data[-1].reshape(len(models), len(xpoints))
    accs = np.log10(accs+1)
    # xpoints = ["500", "1k", "2k", "5k", "10k", "20k"]
    xpoints = "20 60 100 200 350".split()
    accs = accs[:, [4,6,8,9,10]]
    filename = "visualization/images/time_k.png"
    barchart(accs, xpoints, "Average degree", "Time(log(s))", filename,
        yticks=([0,2,4], ["0", "2", "4"]), add_Legend=True)


def draw_density_k():
    print("density")
    xpoints = "4 10 100 200 350 500 700 900 999".split()
    text = open("tests/raw-data/timek.txt").read()

    data = get_info(text)
    accs = data[0].reshape(len(models), len(xpoints))
    xpoints = "4 10 100 500 900 999".split()
    accs = accs[:,[0,1,2,5,7,8]]

    filename = "visualization/images/new_density.png"
    line_chart(accs, xpoints, "k", "Accuracy", filename=filename)


def draw_components():
    print("components")
    xpoints = "1 2 3 4 5 6".split()
    text = open("tests/raw-data/components.txt").read()

    data = get_info(text)
    accs = data[0].reshape(len(models), len(xpoints))
    # xpoints = "4 10 100 500 900 999".split()
    # accs = accs[:,[0,1,2,5,7,8]]

    filename = "visualization/images/components.png"
    line_chart(accs, xpoints, "Connected components", "Accuracy", filename=filename)

def draw_deledges(name, legend):
    print("Del_edges")
    xpoints = [0, 0.01, 0.5, 0.1, 0.2]
    text = "tests/raw-data/de" + name + ".txt"
    data = get_info(text)
    # line_chart(data, xpoints, "Edges removal ratio", "Accuracy",
    #            filename="visualization/images/draw_deledges_{}.png".format(name), models=models, add_Legend=legend)
    barchart(data, xpoints, "Edges removal ratio", "Accuracy",
                filename="visualization/images/draw_deledges_{}.png".format(name), add_Legend=legend)


def draw_delns(name, legend):
    print("Del_nodes")
    xpoints = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    text = "tests/raw-data/delnodes_" + name + ".txt"
    data = get_info(text)
    line_chart(data, xpoints, "Nodes removal ratio", "Accuracy",
               filename="visualization/images/draw_delenodes_{}.png".format(name), models=models, add_Legend=legend)
    # barchart(data, xpoints, "Nodes removal ratio", "Accuracy",
    #          filename="visualization/images/draw_delenodes_{}.png".format(name), add_Legend=legend)

def draw_feats(name, legend):
    print("change feats")
    xpoints = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    text = "tests/raw-data/change_feat_" + name + ".txt"
    data = get_info(text)
    # line_chart(data, xpoints, "Features changing ratio", "Accuracy", 
    #             filename="visualization/images/draw_changefeats_{}.png".format(name), models=feat_models, add_Legend=legend)
    barchart(data, xpoints, "Features changing ratio", "Accuracy",
             filename="visualization/images/draw_changefeats_{}.png".format(name), add_Legend=legend)



def draw_numwalks(legend):
    print("draw_numwalks")
    xpoints = [1, 50, 100, 500, "1k", "2k"]
    text = "tests/raw-data/num_walks.txt"
    data = get_info(text)
    line_chart(data, xpoints, "Number of walks", "Accuracy",
                filename="visualization/images/draw_numwalks.png", models=["FDeep"], add_Legend=legend)

def draw_walklen(legend):
    print("draw_walklen")
    xpoints = [1, 2, 3, 5, 10, 20]
    text = "tests/raw-data/walk_len.txt"
    data = get_info(text)
    line_chart(data, xpoints, "Walk length", "Accuracy",
                filename="visualization/images/walk_len.png", models=["FDeep"], add_Legend=legend)

def draw_windowsize(legend):
    print("draw_windowsize")
    xpoints = [1, 2, 3]
    text = "tests/raw-data/window_size.txt"
    data = get_info(text)
    line_chart(data, xpoints, "window size", "Accuracy",
                filename="visualization/images/window_size.png", models=["FDeep"], add_Legend=legend)


def draw_number_of_content_neighbors(legend):
    print("num content neighbors")
    xpoints = [0, 1, 2, 5, 10, 20, 50]
    text = "tests/raw-data/num_content_neighbors.txt"
    data = get_info(text)
    line_chart(data, xpoints, "number of content neighbors", "Accuracy",
                filename="visualization/images/num_content_neighbors.png", models=["FDeep"], add_Legend=legend)




# models = ["RAN", "RAN_so", "REGAL", "FINAL","BigAlign", "DeepLink"]
# models = ["NAWAL", "REGAL", "FINAL", "BigAlign"]
# models = ["NAWAL", "PALE", "DeepLink"]
# feat_models = ["RAN", "REGAL", "FINAL", "BigAlign"]
# models = feat_models
models_extra = ["REGAL","FINAL","IsoRank","BigAlign","PALE","IONE","DeepLink","FAN"]
models = ["REGAL","FINAL","IsoRank","BigAlign","PALE","IONE","DeepLink"]

# legend = False
# draw_numwalks(legend)
# draw_walklen(legend)
# draw_windowsize(legend)
# draw_number_of_content_neighbors(legend)

# for name in ['ppi', 'econ', 'bn']:
#     legend = False
#     if name == 'econ':
#         legend = True
#     # draw_deledges(name, legend)
#     # draw_delns(name, legend)
#     draw_feats(name, legend)


# get_info('raw-data/semi_regal.txt')
# draw_deledges()
# draw_delns()
# draw_feats()

# draw_realdata()
# draw_semi_regal()
# draw_delnodes()
# draw_density(models=models)
# draw_connectivity("seed123-del01")
# draw_density("seed234-del01")
# draw_connectivity("seed234-del01")
# for i in ["2", "5", "8"]:
#     draw_density("seed123-del{}".format(i), add_FAN=True)
#     draw_connectivity("seed123-del{}".format(i), add_FAN=True)
# draw_changefeats()

# draw_density_k()
# draw_time()
# draw_timek()

# draw_components()
# import json

# ps = ['2', '3', '4', '5', '6', '7']
# name = "fully-synthetic-small-world-n1000-k10-p"


# final_accs = []
# for model in models:
#     path = "output/" + model + "/components/seeds.json"
#     with open(path) as f:
#         data = json.load(f)
#     accps = []
#     for p in ps:
#         data_ele = data[name + p]
#         accs= []
#         for key in data_ele.keys():
#             if data_ele[key]['acc'] is not None:
#                 accs.append(data_ele[key]['acc'])
#         accps.append(np.mean(accs))
#     final_accs.append(accps)
# final_accs = np.array(final_accs)
# print(final_accs)

# line_chart(final_accs, ps, "p", "Accuracy", filename="visualization/images/connectivity.png", models=models, add_Legend=False)



# ps = ['1', '2', '3', '4', '5', '6']
# name = "fully-synthetic-connected-components-small-world-n1000-k10-p5-nc"


# final_accs = []
# for model in models:
#     path = "output/" + model + "/components/seeds.json"
#     with open(path) as f:
#         data = json.load(f)
#     accps = []
#     for p in ps:
#         data_ele = data[name + p]
#         accs= []
#         for key in data_ele.keys():
#             if data_ele[key]['acc'] is not None:
#                 accs.append(data_ele[key]['acc'])
#         accps.append(np.mean(accs))
#     final_accs.append(accps)
# final_accs = np.array(final_accs)
# print(final_accs)

# line_chart(final_accs, ps, "Connected components", "Accuracy", filename="visualization/images/components.png", models=models, add_Legend=True)