import numpy as np

from Solvers.IterativeHardThresholding import iterative_hard_thresholding


def main():
    X = np.array([[0.1, 0.2], [0.3, 0.3]])
    beta = np.array([1, 2])
    init_y = np.array([0, 0])
    y = iterative_hard_thresholding(X, beta, init_y, 1, 10)
    print(y)


if __name__ == "__main__":
    main()
