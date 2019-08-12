import numpy as np
# import scipy.sparse.csr_matrix
import pdb

def greedy_match(S):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:,:k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
    return result

# def get_top_k(S, id2idx_source, id2idx_target, k=1):
#     """
#     S: scores, numpy array of shape (M, N) where M is the number of source nodes,
#         N is the number of target nodes
#     id2idx_source: id_map of source dataset
#     id2idx_target: id_map of target dataset
#     k: number of predicted elements to return
#     """
#     top = np.argsort(-S)[:,:k]
#     # convert top k from index to id
#     idx2id_src = {v:k for k,v in id2idx_source.items()}
#     idx2id_trg = {v:k for k,v in id2idx_target.items()}

#     result = {}
#     for idx, target_elms in enumerate(top):
#         result[idx2id_src[idx]] = []
#         for elm in target_elms:
#             result[idx2id_src[idx]].append(idx2id_trg[elm])
#     return result # source - top_k_of_target

# def greedy_match(S, id2idx_source, id2idx_target):
#     """
#     :param S: Scores matrix, shape MxN where M is the number of source nodes,
#         N is the number of target nodes.
#     :param id2idx_source:
#     :param id2idx_target:
#     :return: A dict, map from source to list of targets.
#     """
#     S = S.T
#     m, n = S.shape
#     x = S.T.flatten()
#     min_size = min([m,n])
#     used_rows = np.zeros((m))
#     used_cols = np.zeros((n))
#     max_list = np.zeros((min_size))
#     row = np.zeros((min_size))  # target indexes
#     col = np.zeros((min_size))  # source indexes

#     ix = np.argsort(-x) + 1

#     matched = 1
#     index = 1
#     while(matched <= min_size):
#         ipos = ix[index-1]
#         jc = int(np.ceil(ipos/m))
#         ic = ipos - (jc-1)*m
#         if ic == 0: ic = 1
#         if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
#             row[matched-1] = ic - 1
#             col[matched-1] = jc - 1
#             max_list[matched-1] = x[index-1]
#             used_rows[ic-1] = 1
#             used_cols[jc-1] = 1
#             matched += 1
#         index += 1
#     # Build dict
#     idx2id_src = {v:k for k,v in id2idx_source.items()}
#     idx2id_trg = {v:k for k,v in id2idx_target.items()}
#     result = {}
#     for i in range(len(row)):
#         result[idx2id_src[col[i]]] = [idx2id_trg[row[i]]]
#     return result

# def compute_accuracy(top_k, groundtruth):
#     matched = 0
#     total_nodes = len(groundtruth.items())
#     for src, trg in groundtruth.items():
#         if trg in top_k[src]:
#             matched+=1
#     return matched*100/total_nodes
