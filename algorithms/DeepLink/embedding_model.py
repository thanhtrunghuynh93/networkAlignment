from gensim.models import Word2Vec
import numpy as np


class DeepWalk:
    def __init__(self, G, id2idx, num_walks=10, walk_len=10, window_size=5, \
                embedding_dim=800, num_cores=8, num_epochs=50, seed=123):
        """
        Parameters
        ----------
        G: networkx Graph
            Graph
        id2idx: dictionary
            dictionary of keys are ids of nodes and values are index of nodes
        num_walks: int
            number of walks per node
        walk_len: int
            length of each walk
        windows_size: int
            size of windows in skip gram model
        embedding_dim: int
            number of embedding dimensions
        num_cores: int
            number of core when train embedding
        num_epochs: int
            number of epochs in embedding
        """
        np.random.seed(seed)

        self.G = G
        self.num_walks = num_walks
        self.walk_len = walk_len
        self.window_size = window_size
        self.id2idx = id2idx
        self.embedding_dim = embedding_dim
        self.num_cores = num_cores
        self.num_epochs = num_epochs
        self.seed = seed


    def get_embedding(self):
        walks = self.run_random_walks()
        import pdb
        pdb.set_trace()
        embedding_model = Word2Vec(walks, size=self.embedding_dim, window=self.window_size,\
                            min_count=0, sg=1, hs=1, workers=self.num_cores, iter=self.num_epochs, seed=self.seed)
        embedding = np.zeros((len(self.G.nodes()), self.embedding_dim))
        for i in range(len(self.G.nodes())):
            embedding[i] = embedding_model.wv[str(i)]
        return embedding



    def run_random_walks(self):
        print("Random walk process")
        walks = []
        for i in range(self.num_walks):
            for count, node in enumerate(self.G.nodes()):
                walk = [str(self.id2idx[node])]
                if self.G.degree(node) == 0:
                    continue
                curr_node = node
                for j in range(self.walk_len):
                    next_node = np.random.choice(self.G.neighbors(curr_node))
                    curr_node = next_node
                    if curr_node != node:
                        walk.append(str(self.id2idx[curr_node]))
                walks.append(walk)
        print("Done walks for", count + 1, "nodes")
        return walks
