import argparse
import os
import subprocess
import sys
import numpy as np
import time

from algorithms.network_alignment_model import NetworkAlignmentModel


class IONE(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, gt_train, epochs=100, dim=100, seed=123):
        """
        :param source_dataset: Dataset object, contains information of source dataset
        :param target_dataset: Dataset object, contains information of target dataset
        :param gt_train: A dict, groundtruth use to train
        :param total_iter: Total training iterations
        :param dim: Embedding dimension
        """
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.gt_train = {}
        with open(gt_train) as file:
            for line in file:
                src, trg = line.split()
                self.gt_train[src] = trg

        # self.total_iter = total_iter
        self.epochs = epochs
        self.dim = dim
        self.temp_dir = "temp/{}".format(time.time())
        self.seed = seed

    def align(self):
        # self._data_to_ione_format()
        train_gt_file, src_edgelist_file, trg_edgelist_file, src_map, trg_map = \
            self._save_data_as_ione_format()
        self._call_java_ione(train_gt_file, src_edgelist_file, trg_edgelist_file)
        alignment_matrix = self._postprocess_alignment_matrix(src_map, trg_map)
        return alignment_matrix

    def _save_data_as_ione_format(self):
        src_edgelist = self.source_dataset.G.edges()
        trg_edgelist = self.target_dataset.G.edges()
        src_nodes = self.source_dataset.G.nodes()
        trg_nodes = self.target_dataset.G.nodes()
        conversion_src = type(self.source_dataset.G.nodes()[0])
        conversion_trg = type(self.target_dataset.G.nodes()[0])
        gt, src_map, trg_map = self._read_gt(self.gt_train, conversion_src, conversion_trg)

        self._add_missing_nodes(src_map, src_nodes)
        self._add_missing_nodes(trg_map, trg_nodes)
        src_new_edgelist = self._create_edgelist(src_edgelist, src_map)
        trg_new_edgelist = self._create_edgelist(trg_edgelist, trg_map)
        new_gt = self._create_gt(gt, src_map, trg_map)
        self.total_iter = self.epochs * max([len(src_new_edgelist), len(trg_new_edgelist)])

        self._make_temp_dir(self.temp_dir)
        src_edgelist_file = os.path.join(self.temp_dir, "src/src.edgelist")
        trg_edgelist_file = os.path.join(self.temp_dir, "trg/trg.edgelist")
        train_gt_file = os.path.join(self.temp_dir, "groundtruth/groundtruth.train")
        self._write_edgelist(src_new_edgelist, src_edgelist_file)
        self._write_edgelist(trg_new_edgelist, trg_edgelist_file)
        self._write_gt(new_gt, train_gt_file)
        return train_gt_file, src_edgelist_file, trg_edgelist_file, src_map, trg_map

    def _read_gt(self, old_gt, conversion_src, conversion_trg):
        """
        :param gt: A dict
        :return:
            gt: gt in list format
            map_src: a dict that map from old node id to new node id for source dataset
            map_trg: a dict that map from old node id to new node id for target dataset
        """
        gt = []
        map_src = {}
        map_trg = {}
        i = 1
        for src, trg in old_gt.items():
            src, trg = conversion_src(src), conversion_trg(trg)
            gt.append([src, trg])
            try:
                # in case src or trg repeat many times in groundtruth
                map_src[src]
                map_src[str(src) + '_2'] = i
            except:
                map_src[src] = i
            try:
                map_trg[trg]
                map_trg[str(trg) + '_2'] = i
            except:
                map_trg[trg] = i
            i += 1
        return gt, map_src, map_trg

    def _add_missing_nodes(self, mapp, nodes):
        i = len(mapp.keys()) + 1
        for node in nodes:
            try:
                mapp[node]
            except:
                # node not in present nodes
                mapp[node] = i
                i += 1
        assert len(set(mapp.values())) == len(mapp.values()), "Missing nodes!"
        assert min(mapp.values()) == 1, "Start idx node wrong!"
        assert max(mapp.values()) == len(mapp.values()), "End idx node wrong!"
        return mapp

    def _create_edgelist(self, edgelist, mapp):
        new_edgelist = []
        for edge in edgelist:
            new_edge = [mapp[edge[0]], mapp[edge[1]]]
            new_edgelist.append(new_edge)
        return new_edgelist

    def _create_gt(self, gt, src_map, trg_map):
        new_gt = []
        for line in gt:
            new_gt.append([src_map[line[0]], trg_map[line[1]]])
            assert src_map[line[0]] == trg_map[line[1]], "Build map wrong!"
        return new_gt

    def _make_temp_dir(self, temp_dir):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if not os.path.exists(temp_dir + "/src/embeddings"):
            os.makedirs(temp_dir + "/src/embeddings")
        if not os.path.exists(temp_dir + "/trg/embeddings"):
            os.makedirs(temp_dir + "/trg/embeddings")
        if not os.path.exists(temp_dir + "/groundtruth"):
            os.makedirs(temp_dir + "/groundtruth")

    def _write_edgelist(self, edgelist, filename):
        with open(filename, "w+") as file:
            for edge in edgelist:
                file.write("{} {}\n".format(*edge))
                if edge[0] != edge[1]:
                    file.write("{} {}\n".format(edge[1], edge[0]))

    def _write_gt(self, gt, filename):
        with open(filename, "w+") as file:
            for line in gt:
                file.write("{}\n".format(line[0]))

    def _call_java_ione(self, train_file, networkx_file, networky_file):
        command = [
            "java",
            "-jar",
            "-Xmx2G",
            "-XX:+UseConcMarkSweepGC",
            "algorithms/IONE/src/IONE.jar",
            networkx_file,
            networky_file,
            train_file,
            "src",
            "trg",
            str(self.total_iter),
            str(self.dim),
            self.temp_dir,
            str(self.seed)
        ]
        print('Command call java: ', ' '.join(map(str, command)))
        process = subprocess.Popen(command, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if err is not None and err.strip():
            err = err.decode('utf8')
            print(err)
            if ("Unable to access jarfile" in err) or ("java.lang.UnsupportedClassVersionError" in err):
                self._compile_java_instruction()
            sys.exit()
        if output is not None:
            output = output.decode('utf8')
            print(output)

    def _compile_java_instruction(self):
        print("You must compile jar file. Use these commands below (requires jdk 1.8.0 or higher):")
        print("cd algorithms/IONE/src")
        print("javac FinalModel/IONE.java")
        print("jar cvfm IONE.jar META-INF/MANIFEST.MF .")
        print("cd ../../..")

    def _postprocess_alignment_matrix(self, src_map, trg_map):
        saved_path = os.path.join(self.temp_dir, "alignment_matrix")
        alignment_matrix = np.zeros((len(src_map.keys()), len(trg_map.keys())))
        reversed_src_map = {v: k for k, v in src_map.items()}
        reversed_trg_map = {v: k for k, v in trg_map.items()}
        with open(saved_path, "r") as file:
            lines = file.readlines()
            trg_ids = list(map(int, lines[0].split(",")[1:]))
            alignment_trg_idxs = [self.target_dataset.id2idx[reversed_trg_map[trg]]
                                  for trg in trg_ids]
            for line in lines[1:]:
                elms = line.split(",")
                src_id = int(elms[0])
                cosins = np.array(list(map(float, elms[1:])), dtype=np.float32)
                alignment_matrix[
                    self.source_dataset.id2idx[reversed_src_map[src_id]],
                    alignment_trg_idxs] = cosins
        return alignment_matrix

