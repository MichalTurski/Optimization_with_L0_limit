import numpy as np

def square_euclidean(vec):
    return np.inner(vec, vec)

def H_operator(vec, dim_thresh):
    #  Select dim_thresh largest elements of vec
    order = np.argsort(-vec)  # "-" because argsort sorts in ascending order
    res = np.zeros_like(vec)
    for i in range(dim_thresh):
        curr_idx = order[i]
        res[curr_idx] = vec[curr_idx]
    return res