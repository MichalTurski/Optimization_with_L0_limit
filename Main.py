import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def plot_losses():
    for i, (X, y) in enumerate(dataset_generator()):
        init_beta = np.random.rand(X.shape[1])
        L0_thresh = int(init_beta.shape[0]/2)
        iter_limit = 1000
        eps = 0.0001
        _, ith_hist = iterative_hard_thresholding(X, y, init_beta, L0_thresh, iter_limit, eps)
        _, dfo_hist = discrete_first_order(X, y, init_beta, L0_thresh, iter_limit, eps)
        _, dfo_mod_hist = discrete_first_order_modified(X, y, init_beta, L0_thresh, iter_limit, eps)
        plot_loss(ith_hist, dfo_hist, dfo_mod_hist, i + 1)


def plot_itrations():
    for i, (X, y) in enumerate(dataset_generator()):
        init_beta = np.random.rand(X.shape[1])
        iter_limit = 1000
        eps = 0.0001
        ith_iters_list = []
        dfo_iters_list = []
        dfo_mod_iters_list = []
        for j in range(50):
            ith_iters = []
            dfo_iters = []
            dfo_mod_iters = []
            for L0_thresh in range(1, int(init_beta.shape[0])):
                _, ith_hist = iterative_hard_thresholding(X, y, init_beta, L0_thresh, iter_limit, eps)
                ith_iters.append(len(ith_hist))
                _, dfo_hist = discrete_first_order(X, y, init_beta, L0_thresh, iter_limit, eps)
                dfo_iters.append(len(dfo_hist))
                _, dfo_mod_hist = discrete_first_order_modified(X, y, init_beta, L0_thresh, iter_limit, eps)
                dfo_mod_iters.append(len(dfo_mod_hist))
            ith_iters_list.append(ith_iters)
            dfo_iters_list.append(dfo_iters)
            dfo_mod_iters_list.append(dfo_mod_iters)
        ith_iters_df = pd.DataFrame(ith_iters_list)
        dfo_iters_df = pd.DataFrame(dfo_iters_list)
        dfo_mod_iters_df = pd.DataFrame(dfo_mod_iters_list)
        ith_iters = ith_iters_df.mean().tolist()
        dfo_iters = dfo_iters_df.mean().tolist()
        dfo_mod_iters = dfo_mod_iters_df.mean().tolist()

        plot_loss(ith_iters, dfo_iters, dfo_mod_iters, i + 1)


if __name__ == "__main__":
    # plot_losses()
    plot_itrations()

