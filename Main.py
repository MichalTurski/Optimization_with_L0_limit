import numpy as np
import matplotlib.pyplot as plt

from Solvers.IterativeHardThresholding import iterative_hard_thresholding
from DatasetGenerator import dataset_generator


def plot_loss(hard_thresholding_hist, nesterov_hist, sample_num):
    plt.plot(hard_thresholding_hist, label='Loss for hard thresholding algorithm')
    plt.plot(nesterov_hist, label="Nesterov's algorithm")  # TODO: change label
    plt.ylim(ymin=0)
    plt.title(f'Loss for sample {sample_num}')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()


def main():
    # X = np.array([[0.1, 0.2], [0.3, 0.3]])
    # beta = np.array([1, 2])
    # init_y = np.array([0, 0])
    # y, hist = iterative_hard_thresholding(X, beta, init_y, 1, 100, 0.06)
    # plot_loss(hist, np.multiply(2 * hist, 2), 1)

    for i, (X, y) in enumerate(dataset_generator()):
        print(f'X shape = {X.shape}, y shape = {y.shape}')
        init_beta = np.random.rand(X.shape[1])
        init_beta = init_beta / np.linalg.norm(init_beta)
        L0_thresh = int(init_beta.shape[0]/2)
        iter_limit = 1000
        eps = 10000
        # y = y[:, 0]  # TODO: remove this line after fix in generator
        _, ith_hist = iterative_hard_thresholding(X, y, init_beta, L0_thresh, iter_limit, eps)
        #TODO: Call Nesterov's algorithm
        nesterov_hist_mock = np.random.rand(len(ith_hist))
        plot_loss(ith_hist, nesterov_hist_mock, i + 1)


if __name__ == "__main__":
    main()
