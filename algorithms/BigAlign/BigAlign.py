import numpy as np
import numpy.matlib as matlib
from scipy.sparse.linalg import svds
from algorithms.network_alignment_model import NetworkAlignmentModel
from input.dataset import Dataset

class BigAlign(NetworkAlignmentModel):
    def __init__(self, src_data, trg_data, lamb=0.01):
        """
        data1: object of Dataset class, contains information of source network
        data2: object of Dataset class, contains information of target network
        lamb: lambda
        """
        self.src_data = src_data
        self.trg_data = trg_data
        self.lamb = lamb

        if (self.src_data.features is not None) and (self.trg_data.features is not None):
            np.random.seed(123)
            self.weight_features = np.random.uniform(size=(self.src_data.features.shape[1],
                3))

        self.N1 = self._extract_features(self.src_data)
        self.N2 = self._extract_features(self.trg_data)


    def _extract_features(self, dataset):
        """
        Preprocess input for unialign algorithms
        """
        n_nodes = len(dataset.G.nodes())
        if dataset.features is not None:
            nodes_degrees = dataset.get_nodes_degrees()
            nodes_clustering = dataset.get_nodes_clustering()
            if dataset.features.shape[1] > 3:
                features = dataset.features.dot(self.weight_features)
            else:
                features = dataset.features
            N = np.zeros((n_nodes, 2+features.shape[1]))
            N[:,0 ] = nodes_degrees
            N[:,1 ] = nodes_clustering
            N[:,2:] = features
        else:
            N = np.zeros((n_nodes,2))
            N[:,0] = dataset.get_nodes_degrees()
            N[:,1] = dataset.get_nodes_clustering()
        return N
        # N = np.zeros((n_nodes, 3))
        # N[:, 0] = dataset.get_nodes_degrees()
        # N[:, 1] = dataset.get_nodes_clustering()
        # if dataset.features is not None:
        #     N[:,2] = dataset.features.argmax(axis=1)
        # return N

    def align(self):
        N1, N2, lamb = self.N1, self.N2, self.lamb
        n2 = N2.shape[0]
        d = N2.shape[1]
        u, s, _ = np.linalg.svd(N1, full_matrices=False)

        # transform s
        S = np.zeros((s.shape[0], s.shape[0]))

        for i in range(S.shape[1]):
            S[i, i] = s[i]
            S[i, i] = 1 / S[i, i] ** 2

        X = N1.T.dot(u).dot(S).dot(u.T)
        Y = lamb / 2 * np.sum(u.dot(S).dot(u.T), axis=0)
        P = N2.dot(X) - matlib.repmat(Y, n2, 1)
        return P.T  # map from source to target
