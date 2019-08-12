import numpy as np
from sklearn.preprocessing import normalize
import os
from copy import deepcopy
import networkx as nx


class Feat2vec:
    """
    A class for encode features on nodes of a network as nodes
    """
    def __init__(self, G, id2idx, feats, num_content_edges=0):
        self.id2idx = deepcopy(id2idx)
        self.feats = deepcopy(feats)
        self.normalize_feats()

        self.struct_G = G
        self.content_G = None
        self.struct_content_G = None
        self.full_G = G.copy()


        self.structure_nodes = G.nodes()
        self.structure_deg = None
        self.content_deg = None
        self.num_nodes = None
        self.num_content_edges = num_content_edges
        self.feat_dict_id = {}

        if feats is not None:
            self.gen_structure_content_edges(self.feats)
            self.gen_content_edges(self.unique_feats, self.num_content_edges, self.feat_id2idx)
            self.id2idx.update(self.feat_id2idx)
            self.content_G = self.create_G(self.content_edge_id)
            self.struct_content_G = self.create_G(self.struct_content_edges)
            self.full_G.add_edges_from(self.content_edge_id)
            self.full_G.add_edges_from(self.struct_content_edges)
        
        self.deg = self.construct_deg(self.full_G, self.id2idx)     
        self.structure_deg = self.deg[:len(self.structure_nodes)]
        if feats is not None:
            self.content_deg = np.zeros(len(self.deg))
            self.content_deg[len(self.structure_deg):] = self.deg[len(self.structure_deg):]
        self.num_nodes = len(self.full_G.nodes())


    def get_full_edges(self):
        edges = [[self.id2idx[edge[0]], self.id2idx[edge[1]]] for edge in self.full_G.edges()]
        edges += [[self.id2idx[edge[1]], self.id2idx[edge[0]]] for edge in self.full_G.edges()]
        return edges

    def get_edges_sets(self):
        struct_edges = self.struct_G.edges()
        content_edges = self.content_G.edges()
        struct_content_edges = self.struct_content_G.edges()
        struct_edges_index = [[self.id2idx[edge[0]], self.id2idx[edge[1]]] for edge in struct_edges]
        struct_edges_index += [[self.id2idx[edge[1]], self.id2idx[edge[0]]] for edge in struct_edges]
        content_edges_index = [[self.id2idx[edge[0]], self.id2idx[edge[1]]] for edge in content_edges]
        content_edges_index += [[self.id2idx[edge[1]], self.id2idx[edge[0]]] for edge in content_edges]
        struct_content_edges_index = [[self.id2idx[edge[0]], self.id2idx[edge[1]]] for edge in struct_content_edges]
        struct_content_edges_index += [[self.id2idx[edge[1]], self.id2idx[edge[0]]] for edge in struct_content_edges]
        return struct_edges_index, content_edges_index, struct_content_edges_index
    

    def weight_function(self, edge0, edge1):
        weight1 = np.log(len(self.struct_G.neighbors(edge0)) + 1) * np.log(len(self.struct_G.neighbors(edge1)) + 1)
        weight2 = np.log(len(self.struct_G.neighbors(edge0)) + len(self.struct_G.neighbors(edge1))) # bestttttttt
        weight3 = np.log(len(self.struct_G.neighbors(edge0)) * len(self.struct_G.neighbors(edge1)) + 1)
        weight4 = np.sqrt(weight1)
        weight5 = np.sqrt(weight2)
        weight6 = np.sqrt(weight3)
        return weight2


    def get_edge_with_weight(self):
        struct_edges = self.struct_G.edges()
        struct_content_edges = self.struct_content_G.edges()
        weight = [self.weight_function(edge[0], edge[1]) for edge in struct_edges]
        sum_weight = np.sum(weight)

        num_edges = len(struct_edges)
        edges = [[self.id2idx[edge[0]], self.id2idx[edge[1]], num_edges * self.weight_function(edge[0], edge[1]) / sum_weight] \
                        for edge in struct_edges]
        edges += [[self.id2idx[edge[1]], self.id2idx[edge[0]], num_edges * self.weight_function(edge[0], edge[1]) / sum_weight] \
                        for edge in struct_edges]
        edges += [[self.id2idx[edge[0]], self.id2idx[edge[1]], 1] for edge in struct_content_edges]
        edges += [[self.id2idx[edge[1]], self.id2idx[edge[0]], 1] for edge in struct_content_edges]
        return edges


    def get_quadra(self):
        quadra = []
        for node in self.struct_G.nodes():
            contexts = self.struct_G.neighbors(node)
            content = self.struct_content_G.neighbors(node)
            content_contexts = self.content_G.neighbors(content[0])
            quadra += [[node, ctx, content[0], ct_ctx] for ctx in contexts for ct_ctx in content_contexts]
        return quadra
                
    def get_struct_edges(self):
        struct_edges = self.struct_G.edges()
        struct_edges_index = [[self.id2idx[edge[0]], self.id2idx[edge[1]]] for edge in struct_edges]
        struct_edges_index += [[self.id2idx[edge[1]], self.id2idx[edge[0]]] for edge in struct_edges]    
        return struct_edges_index
    


        
    def get_nodes_and_context(self, struct_unique_nodes_id):
        struct_nodes = []
        struct_context = []
        for i in range(len(struct_unique_nodes_id)):
            neighbors_i = self.struct_G.neighbors(struct_unique_nodes_id[i])
            center = [struct_unique_nodes_id[i]] * len(neighbors_i)
            struct_nodes += center
            struct_context += neighbors_i
        return struct_nodes, struct_context
    
    def get_content_nodes(self, struct_unique_nodes_id):
        content_unique_nodes = []
        content = []
        content_context = []
        content_simi = []
        for i in range(len(struct_unique_nodes_id)):
            struct_node = struct_unique_nodes_id[i]
            content_node = self.struct_content_G.neighbors(struct_node)[0]
            content_unique_nodes.append(content_node)
            if len(self.content_G.nodes()) > 0:
                content_context_i = self.content_G.neighbors(content_node)
                content += [content_node] * len(content_context_i)
                content_context += content_context_i
                for j in range(len(content_context_i)):
                    content_simi.append(float(np.sum(self.feat_dict_id[content_node] * self.feat_dict_id[content_context_i[j]])))
        return content_unique_nodes, content, content_context, content_simi
    
    def get_feature_all_nodes(self):
        feats = np.zeros((len(self.full_G.nodes()), self.feats.shape[1]))
        feats[:len(self.feats)] = self.feats
        for id in self.feat_dict_id.keys():
            feats[self.id2idx[id]] = self.feat_dict_id[id]
        return feats

    def get_idx2id(self, id2idx = None):
        """
        return idx2id based on id2idx
        """
        if id2idx is None:
            return {v:k for k, v in self.id2idx.items()}
        return {v:k for k, v in id2idx.items()}
    
    def normalize_feats(self):
        """
        feature will be nomalized such that its magnitude equals to 1
        """
        if self.feats is not None:
            self.feats = normalize(self.feats, axis=1)



        

    def gen_structure_content_edges(self, feats):
        """
        input: features
        return: - list of unique features
                - new_s-c edges id
                - new_s-c edges idx
                - feat_dict_id: content of feat with id
                - feat_dict_idx: content of feat with idx
        """
        max_len = 0
        last_node = None
        for node in self.struct_G.nodes():
            node = str(node)
            if len(node) > max_len:
                max_len = len(node)
                last_node = node

        idx2id = self.get_idx2id()

        unique_feats = [feats[0]]
        feat_dict_id = {last_node + '0': feats[0]} # feat_id: feat_value
        feat_dict_idx = {len(self.struct_G.nodes()): feats[0]} # feat_idx: feat_value
        feat_id2idx = {last_node + '0': len(self.struct_G.nodes())}
        new_edges_id = [[idx2id[0], last_node + '0']]
        new_edges_idx = [[0, len(self.struct_G.nodes())]]
        count_unique_feat = 1
        for i in range(len(feats)):
            equal = 0
            for j in range(len(unique_feats)):
                feat_idx = j + len(self.struct_G.nodes())
                if np.array_equal(unique_feats[j], feats[i]):
                    new_edges_id.append([idx2id[i], last_node + str(j)])
                    new_edges_idx.append([i, feat_idx])
                    equal = 1
                    break
            if equal == 0:
                new_feat_node_id = last_node + str(count_unique_feat)
                new_feat_node_idx = len(self.struct_G.nodes()) + count_unique_feat
                feat_id2idx[new_feat_node_id] = new_feat_node_idx
                feat_dict_id[new_feat_node_id] = feats[i]
                feat_dict_idx[new_feat_node_idx] = feats[i]
                new_edges_id.append([idx2id[i], new_feat_node_id])
                new_edges_idx.append([i, new_feat_node_idx])
                unique_feats.append(feats[i])
                count_unique_feat += 1
        self.unique_feats = unique_feats
        print("number of unique feature is: ", len(self.unique_feats))
        self.struct_content_edges = new_edges_id
        self.feat_id2idx = feat_id2idx
        self.feat_dict_id = feat_dict_id
        return unique_feats, new_edges_id, new_edges_idx, feat_id2idx, feat_dict_id, feat_dict_idx
    

    def gen_content_edges(self, unique_feats, k, feat_id2idx):
        """
        generate content edges based on topk cosin similarity between content vectors
        input:  unique_feats: list of unique feature
                k: number of neighbor per node
                feat_id2idx
        return: content_edge_id, content_edges_idx
        """
        idxs = list(feat_id2idx.values())
        min_idx = min(idxs)
        feats_array = np.array(unique_feats)
        feats_simi_matrix = np.matmul(feats_array, feats_array.T)
        sorted_simi_matrix = np.argsort(feats_simi_matrix, axis=1)
        content_edges_idx = []
        content_edges_id = []
        feat_idx2id = self.get_idx2id(feat_id2idx)
        for i in range(len(unique_feats)):
            for j in range(k):
                source_idx = i + min_idx
                target_idx = int(sorted_simi_matrix[i, -(j + 2)]) + min_idx
                source_id = feat_idx2id[source_idx]
                target_id = feat_idx2id[target_idx]
                
                content_edges_idx.append([source_idx, target_idx])
                content_edges_id.append([source_id, target_id])
        self.content_edge_id = content_edges_id
        return content_edges_id, content_edges_idx



    def construct_deg(self, G, id2idx):
        """

        """
        degree = np.zeros(len(G.nodes()))
        for node in G.nodes():
            degree[id2idx[node]] = len(G.neighbors(node))
        return degree
        


    def map_feats(self, source_features, target_features, source_feats_id2idx, target_feats_id2idx, min_idx1, min_idx2):
        """
        input: 
            - source unique features list
            - target unique features list
            - source features id2idx
            - target features id2idx
        output: mapf: dictionary of source feat idx corresponding to target feat idx
        """
        source_feats_idx2id = self.get_idx2id(source_feats_id2idx)
        target_feats_idx2id = self.get_idx2id(target_feats_id2idx)
        idxs1 = list(source_feats_id2idx.values())
        idxs2 = list(target_feats_id2idx.values())
        map_feats_id = {}
        map_feats_idx = {}
        for i in range(len(source_features)):
            for j in range(len(target_features)):
                if np.array_equal(source_features[i], target_features[j]):
                    idx1 = min_idx1 + i
                    idx2 = min_idx2 + j
                    map_feats_idx[idx1] = idx2
                    map_feats_id[source_feats_idx2id[idx1]] = target_feats_idx2id[idx2]
                    break
        return map_feats_id, map_feats_idx

    def update_train_dict(self, old_train_dict, train_dict_to_append, out_dir=None, filename=None):
        """
        update train_dict based on traindict to append
        if out_dir is specified, then save the new traindict to path
        """
        if out_dir is None:
            return old_train_dict

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        old_train_dict.update(train_dict_to_append)
        print(filename)

        with open("{0}/{1}".format(out_dir, filename), 'wt') as f:
            for key in old_train_dict:
                f.write("%s %s\n" % (key, old_train_dict[key]))
            f.close()
        return old_train_dict


    # def update_G(self, G, new_edges):
    #     """
    #     return new_graph and new_id2idx
    #     """
    #     self.G.add_edges_from(new_edges)  
    #     return G

    def create_G(self, edges):
        G = nx.Graph()
        G.add_edges_from(edges)
        return G
    
    def to_word2vec_format(self, val_embeddings, nodes, out_dir, filename, dim, id2idx, pref=""):
        val_embeddings = val_embeddings.cpu().detach().numpy()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open("{0}/{1}".format(out_dir, filename), 'w') as f_out:
            f_out.write("%s %s\n"%(len(nodes), dim))
            for node in nodes:
                txt_vector = ["%s" % val_embeddings[int(id2idx[node])][j] for j in range(dim)]
                f_out.write("%s%s %s\n" % (pref, node, " ".join(txt_vector)))
            f_out.close()
        print("emb has been saved to: {0}/{1}".format(out_dir, filename))

        
