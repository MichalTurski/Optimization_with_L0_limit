import numpy as np


def square_euclidean(vec):
    return np.inner(vec, vec)


def iterative_hard_thresholding(X, y, beta_init, dim_thresh, iter_limit, eps):
    X_t = np.transpose(X)
    beta = beta_init
    loss_hist = []
    for i in range(iter_limit):
        H_operator_arg = beta + np.dot(X_t, y - np.dot(X, beta))
        beta = H_operator(H_operator_arg, dim_thresh)
        diff = y - np.dot(X, beta)
        loss = square_euclidean(diff)
        loss_hist.append(loss)
        if loss < eps:
            break
    return beta, loss_hist


def H_operator(vec, dim_thresh):
    #  Select dim_thresh largest elements of vec
    order = np.argsort(-vec)  # "-" because argsort sorts in ascending order
    res = np.zeros_like(vec)
    for i in range(dim_thresh):
        curr_idx = order[i]
        res[curr_idx] = vec[curr_idx]
    return res
