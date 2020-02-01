import numpy as np
import matplotlib.pyplot as plt

from Solvers.IterativeHardThresholding import iterative_hard_thresholding
from Solvers.DiscreteFirstOrder import discrete_first_order, discrete_first_order_modified
from DatasetGenerator import dataset_generator


def plot_loss(hard_thresholding_hist, dfo_hist, dfo_mod_hist, sample_num):
    plt.plot(hard_thresholding_hist, label='Hard thresholding algorithm')
    plt.plot(dfo_hist, label="DFO algorithm")
    plt.plot(dfo_mod_hist, label="DFO modified algorithm")
    plt.ylim(ymin=0)
    # plt.yscale("log")
    plt.title(f'Loss for dataset {sample_num}')
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.legend()
    plt.show()


def main():
    for i, (X, y) in enumerate(dataset_generator()):
        init_beta = np.random.rand(X.shape[1])
        L0_thresh = int(init_beta.shape[0]/2)
        iter_limit = 1000
        eps = 0.0001
        _, ith_hist = iterative_hard_thresholding(X, y, init_beta, L0_thresh, iter_limit, eps)
        _, dfo_hist = discrete_first_order(X, y, init_beta, L0_thresh, iter_limit, eps)
        _, dfo_mod_hist = discrete_first_order_modified(X, y, init_beta, L0_thresh, iter_limit, eps)
        plot_loss(ith_hist, dfo_hist, dfo_mod_hist, i + 1)


if __name__ == "__main__":
    main()
