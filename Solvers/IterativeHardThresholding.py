import numpy as np
from Solvers.utils import H_operator, square_euclidean


def iterative_hard_thresholding(X, y, beta_init, dim_thresh, iter_limit, eps):
    X_t = np.transpose(X)
    beta = H_operator(beta_init, dim_thresh)
    loss_hist = [square_euclidean(y - np.dot(X, beta))]
    for i in range(iter_limit):
        H_operator_arg = beta + np.dot(X_t, y - np.dot(X, beta))
        beta = H_operator(H_operator_arg, dim_thresh)
        diff = y - np.dot(X, beta)
        loss = square_euclidean(diff)
        loss_hist.append(loss)
        if loss_hist[-2] - loss < eps and i > 5:
            break
    return beta, loss_hist
