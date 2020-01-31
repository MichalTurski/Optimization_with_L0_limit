import numpy as np
from Solvers.utils import H_operator, square_euclidean


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
