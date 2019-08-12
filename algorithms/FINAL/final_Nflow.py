import numpy as np
from sklearn.preprocessing import normalize
from algorithms.dataset import Dataset, load_mat, get_H, load_groundtruth
from evaluation.matcher import get_top_k, compute_accuracy, greedy_match
import numpy as np
from numpy import inf, nan
from copy import deepcopy
import argparse
from numpy import inf
import os
import scipy.sparse as sparse
from scipy.linalg import eigh
import scipy

def parse_args():
    parser = argparse.ArgumentParser(description="IsoRank")
    parser.add_argument('--prefix1',             default="/dataspace/graph/pale_facebook/random_clone/sourceclone,alpha_c=0.9,alpha_s=0.9/pale_facebook")
    parser.add_argument('--prefix2',             default="/dataspace/graph/pale_facebook/random_clone/sourceclone,alpha_c=0.9,alpha_s=0.9/pale_facebook")
    parser.add_argument('--groundtruth',        default=None)
    parser.add_argument('--H', default=None)
    parser.add_argument('--alpha',          default=0.82, type=float)
    parser.add_argument('--r', default=1, type=int)
    parser.add_argument('--k', default=1, type=int)
    return parser.parse_args()

def FINAL_NFLOW(A1, A2, N1, N2, H, alpha, r):
    """
    Description:
    This function computes the similarity matrix S, for the scenario that
    only categorical/numerical node attributes are available in two
    networks.
    To explain, categorical node attributes can be taken examples as gender
    (including two different attributes, male and female), locations
    (including different countries, and so on). Numerical node attributes
    can be the number of papers published in different venues by an author.

    Input:
        - A1, A2: adjacency matrices of two networks G1 and G2;
        - N1, N2: Node attributes matrices, N1 is an n1*K matrix, N2 is an n2*K
        matrix, each row is a node, and each column represents an
        attribute. If the input node attributes are categorical, we can
        use one hot encoding to represent each node feature as a vector.
        And the input N1 and N2 are still n1*K and n2*K matrices.
        E.g., for node attributes as countries, including USA, China, Canada, 
        if a user is from China, then his node feature is (0, 1, 0).
         
        - H: an n2*n1 prior node similarity matrix, e.g., degree similarity. H
        should be normalized, e.g., sum(sum(H)) = 1.
        - alpha: a parameter that controls the importance of the consistency
        principles, that is, 1-alpha controls the importance of prior
        knowledge.
        - r: the rank of the low-rank approximations on matrices A1 and A2.
    Output:
        - S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
        x in A2 is aligned to node-y in A1
    """
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    K = N1.shape[1]

    # Normalize node feature vectors
    N1 = normalize(N1)
    N2 = normalize(N2)

    N = np.zeros(n1 * n2)

    for k in range(K):
        N += np.kron(N1[:, k], N2[:, k])
    N = N.reshape((N.shape[0], 1))

    # compute the Kronecker degree vector
    d = np.zeros(n1 * n2)
    for k in range(K):
        d += np.kron(np.dot(A1, N1[:, k]), np.dot(A2, N2[:, k]))
    d = d.reshape((d.shape[0], 1))

    DD = N*d
    D = 1./np.sqrt(DD)
    D[DD == 0] = 0
    
    # Low-rank approximation on A1 and A2 by eigenvalue decomposition
    Lambda1, U1 = scipy.linalg.eig(A1) 
    Lambda2, U2 = scipy.linalg.eig(A2)
    U1 = U1.real
    U2 = U2.real
    Lambda1 = Lambda1.real
    Lambda2 = Lambda2.real
    am1 = np.absolute(Lambda1).argsort()[-r:][::-1]
    am2 = np.absolute(Lambda2).argsort()[-r:][::-1]
    
    Lambda1 = Lambda1[am1]
    Lambda2 = Lambda2[am2]
    U1 = U1[:, am1]
    U2 = U2[:, am2]

    # compute the matrix \Lambda by the equations in the paper
    U = np.kron(U1, U2)
    eta = np.dot(U.T, N*D*U)

    Lambda = np.kron(np.diag(1./Lambda2), np.diag(1./Lambda1)) - alpha * eta

    h = H.flatten('F')
    h = h.reshape((h.shape[0], 1))
    x = alpha*(1-alpha)*N*D*h
    # compute the approximate closed-form solution
    s = (1-alpha)*h + D*N*np.dot(U, np.linalg.lstsq(Lambda, np.dot(U.T, x), rcond=-1)[0])
    S = s.reshape((n1, n2))
    return S


if __name__ == "__main__":
    args = parse_args()
    print(args)

    data1 = Dataset(args.prefix1)
    data2 = Dataset(args.prefix2)
    A1 = data1.adjacency # (3096x3096)
    A2 = data2.adjacency # (1118x1118)
    N1 = data1.features # (3096x538)
    N2 = data2.features # (1118x538)
    groundtruth = load_groundtruth(args.groundtruth)
    H = get_H(args.H, data1, data2) # (1118, 3096)

    S = FINAL_NFLOW(A1, A2, N1, N2, H, args.alpha, args.r)

    matched = get_top_k(S, data1.id2idx, data2.id2idx, k=args.k)
    acc = compute_accuracy(matched, groundtruth)
    print("Top-k accuracy (%d pairs): %.2f%%"%(len(matched.items()),acc))
    matched = greedy_match(S, data1.id2idx, data2.id2idx)
    acc = compute_accuracy(matched, groundtruth)
    print("Greedy match accuracy (%d pairs): %.2f%%"%(len(matched.items()),acc))
