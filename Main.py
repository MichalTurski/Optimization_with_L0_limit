#%%
import numpy as np


#%% ITERATIVE HARD THRESHOLDING
def iterative_hard_thresholding(X, beta, y_init, dim_thresh, iter_limit):
    X_t = np.transpose(X)
    y = y_init
    for i in range(iter_limit):
        H_operator_arg = y + np.dot(X_t, beta - np.dot(X, y))
        y = H_operator(H_operator_arg, dim_thresh)
    return y


def H_operator(vec, dim_thresh):
    #  Select dim_thresh largest elements of vec
    order = np.argsort(-vec)  # "-" because argsort sorts in ascending order
    res = np.zeros_like(vec)
    for i in range(dim_thresh):
        curr_idx = order[i]
        res[curr_idx] = vec[curr_idx]
    return res


#%%
X = np.array([[0.1, 0.2], [0.3, 0.3]])
beta = np.array([1, 2])
init_y = np.array([0, 0])
y = iterative_hard_thresholding(X, beta, init_y, 1, 10)

print(y)
