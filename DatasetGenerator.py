import numpy as np


def create_dataset(cov_matrix, beta, n=500, snr=1):
    p = cov_matrix.shape[0]
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, cov_matrix, size=n)
    X = X / np.linalg.norm(X)
    X_beta = np.matmul(X, beta)
    epsilon = np.random.normal(0, scale=np.sqrt(np.var(X_beta)/snr))
    y = X_beta + epsilon
    return X, y


def dataset_generator():
    # Page 836 Bertsimas
    # Set 1.
    p = 50
    k = 5
    val = 0.5
    cov_matrix = np.empty(shape=[p, p])
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            cov_matrix[i, j] = val ** np.abs(i-j)

    beta = np.zeros(p)
    for i in range(k):
        beta[i] = 1
    yield create_dataset(cov_matrix, beta)

    # Set 2.
    p = 50
    k = 5
    cov_matrix = np.eye(p)
    beta = np.zeros(p)
    for i in range(k):
        beta[i] = 1
    yield create_dataset(cov_matrix, beta)

    # Set 3.
    p = 50
    k = 10
    cov_matrix = np.eye(p)
    beta = np.zeros(p)
    for i in range(k):
        beta[i] = 0.5 + 9.5 * i/k
    yield create_dataset(cov_matrix, beta)

    # Set 4.
    p = 50
    cov_matrix = np.eye(p)
    beta_val = np.array([-10, -6, -2, 2, 6, 10, *[0]*(p-6)])
    yield create_dataset(cov_matrix, beta_val)
