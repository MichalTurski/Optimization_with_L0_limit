import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Solvers.IterativeHardThresholding import iterative_hard_thresholding
from Solvers.DiscreteFirstOrder import discrete_first_order, discrete_first_order_modified
from DatasetGenerator import dataset_generator


def universal_line_plot(tile, x_label, y_label, x_offset, hard_thresholding, dfo, dfo_mod):
    my_dpi = 96
    plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    plt.plot(list(range(x_offset, len(hard_thresholding)+x_offset)), hard_thresholding,
             label='Hard thresholding algorithm')
    plt.plot(list(range(x_offset, len(dfo)+x_offset)), dfo, label="DFO algorithm")
    plt.plot(list(range(x_offset, len(dfo_mod)+x_offset)), dfo_mod, label="DFO modified algorithm")
    plt.ylim(ymin=0)
    # plt.yscale("log")
    plt.title(tile)
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.legend()
    plt.show()


def plot_loss(hard_thresholding_hist, dfo_hist, dfo_mod_hist, dataset_num):
    universal_line_plot(f'Loss for dataset {dataset_num}', 'Iteration', 'Loss', 0,
                        hard_thresholding_hist, dfo_hist, dfo_mod_hist)


def plot_thresh_to_iters(hard_thresholding_iters, dfo_iters, dfo_mod_iters, dataset_num):
    universal_line_plot(f'Iterations number as a threshold function for dataset {dataset_num}', 'Threshold',
                        'Iterations', 1, hard_thresholding_iters, dfo_iters, dfo_mod_iters)


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


def plot_iterations():
    for i, (X, y) in enumerate(dataset_generator()):
        iter_limit = 1000
        eps = 0.0001
        ith_iters_list = []
        dfo_iters_list = []
        dfo_mod_iters_list = []
        for j in range(50):
            init_beta = np.random.rand(X.shape[1])
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

        plot_thresh_to_iters(ith_iters, dfo_iters, dfo_mod_iters, i + 1)


def coefs_barplot(ith_coefs, dfo_coefs, dfo_mod_coefs, dataset_num):
    x = np.arange(len(ith_coefs))  # the label locations
    width = 0.9  # the width of the bars

    my_dpi = 96
    fig, ax = plt.subplots(figsize=(1200 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    rects1 = ax.bar(x - width / 3, ith_coefs, width/3, label='Hard thresholding algorithm')
    rects2 = ax.bar(x, dfo_coefs, width/3, label='DFO algorithm')
    rects3 = ax.bar(x + width / 3, dfo_mod_coefs, width/3, label='DFO modified algorithm')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Coefficient value', fontsize=12)
    ax.set_xlabel('Coefficient number', fontsize=12)
    ax.set_title(f'Coefficients comparison for dataset {dataset_num}')
    ax.set_xticks(np.arange(0, len(ith_coefs), 5))
    # ax.set_xticklabels(labels)
    ax.legend()
    plt.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    # plt.savefig('my_fig.png', dpi=my_dpi)
    plt.show()
    pass


def plot_coefficients():
    for i, (X, y) in enumerate(dataset_generator()):
        init_beta = np.random.rand(X.shape[1])
        L0_thresh = int(init_beta.shape[0]/2)
        iter_limit = 1000
        eps = 0.0001
        ith_coefs,  _ = iterative_hard_thresholding(X, y, init_beta, L0_thresh, iter_limit, eps)
        dfo_coefs, _ = discrete_first_order(X, y, init_beta, L0_thresh, iter_limit, eps)
        dfo_mod_coefs, _ = discrete_first_order_modified(X, y, init_beta, L0_thresh, iter_limit, eps)
        coefs_barplot(ith_coefs, dfo_coefs, dfo_mod_coefs, i + 1)


if __name__ == "__main__":
    plot_losses()
    # plot_iterations()
    # plot_coefficients()
