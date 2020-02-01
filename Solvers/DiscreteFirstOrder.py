import numpy as np
from Solvers.utils import H_operator, square_euclidean


def discrete_first_order_modified(X, y, beta_init, dim_thresh, iter_limit, eps, l_gain=100):
    X_t = np.transpose(X)
    beta = H_operator(beta_init, dim_thresh)
    loss_hist = [square_euclidean(y - np.dot(X, beta))]
    L = l_gain * max(np.linalg.eigvals(np.dot(X_t, X)))
    for i in range(iter_limit):
        H_operator_arg = beta + L * np.dot(X_t, y - np.dot(X, beta))
        eta = H_operator(H_operator_arg, dim_thresh)
        etabetax = np.dot(X, eta - beta)
        lamb = np.dot(np.transpose(etabetax), y - np.dot(X, beta)) / square_euclidean(etabetax)
        beta = lamb * eta + (1 - lamb) * beta
        diff = y - np.dot(X, beta)
        loss = square_euclidean(diff)
        loss_hist.append(loss)
        if loss_hist[-2] - loss < eps and i > 5:
            break
    return beta, loss_hist


def discrete_first_order(X, y, beta_init, dim_thresh, iter_limit, eps, l_gain=100):
    X_t = np.transpose(X)
    beta = H_operator(beta_init, dim_thresh)
    loss_hist = [square_euclidean(y - np.dot(X, beta))]
    L = l_gain * max(np.linalg.eigvals(np.dot(X_t, X)))
    for i in range(iter_limit):
        H_operator_arg = beta + L * np.dot(X_t, y - np.dot(X, beta))
        beta = H_operator(H_operator_arg, dim_thresh)
        diff = y - np.dot(X, beta)
        loss = square_euclidean(diff)
        loss_hist.append(loss)
        if loss_hist[-2] - loss < eps and i > 5:
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
