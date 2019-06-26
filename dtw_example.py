"""
Dynamic Time Warping (DTW) distance between two time series.

DTW implentations source:
https://github.com/pierre-rouanet/dtw

tslearn package:
https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.metrics.html

"""
import os
import numpy as np
import dtw
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sequence(n, curve='sine', seed=123, noise=True):
    n_max = 1e-3
    if n < 1:
        logger.warning("Sequence length is 0. Returning empty array")
        return np.array([])

    np.random.seed(seed=seed)
    ni = nf = 0
    if noise:
        a = (0.15 - 0.1) * np.random.rand() + 0.1
        ni = max(1, int(np.floor(a * n)))
        a = (0.15 - 0.1) * np.random.rand() + 0.1
        nf = max(1, int(np.floor(a * n)))

    m = n - ni - nf
    if curve == 'sine':
        x = np.sin(((2 * np.pi) / m) * np.arange(m))
    elif curve == 'cosine':
        x = np.cos(((2 * np.pi) / m) * np.arange(m))
    else:
        raise ValueError("Invalid value '{}' for parameter 'curve'".format(curve))

    if noise:
        x = x + n_max * np.random.rand(x.shape[0])
        # Append noise values close to 0 of length `noise_interval` at the start and end of the sequence
        if ni > 0:
            xi = n_max * np.random.rand(ni)
            x = np.concatenate([xi, x])

        if nf > 0:
            xf = n_max * np.random.rand(nf)
            x = np.concatenate([x, xf])

    return x


def main():
    # Generate the sequences
    m = 20
    n = 25
    x1 = generate_sequence(m, curve='sine')
    x2 = generate_sequence(n, curve='sine')

    # Two small example sequences for testing
    # x1 = np.array([2, 4, 6, 3, 1, 2], dtype=np.float)
    # x2 = np.array([3, 4, 6, 3, 1, 2, 2, 1, 1], dtype=np.float)

    # Calculate the DTW distance between the sequences
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(x1[:, np.newaxis], x2[:, np.newaxis],
                                                                   dist='cityblock')
    path = zip(path[0], path[1])
    logger.info("DTW distance = %.6f", dist)

    # Generate some plots
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(x1.shape[0]), x1, linestyle='--', color='r', marker='+', markersize=4)
    ax1.plot(np.arange(x2.shape[0]), x2, linestyle='--', color='g', marker='+', markersize=4)
    # ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(np.arange(x1.shape[0]), x1, linestyle='--', color='r', marker='+', markersize=4)
    ax1.plot(np.arange(x2.shape[0]), x2, linestyle='--', color='g', marker='+', markersize=4)
    for tup in path:
        ax1.plot([tup[0], tup[1]], [x1[tup[0]], x2[tup[1]]], linestyle=':', color='grey', marker=',')

    ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')
    plt.plot()
    plot_file = 'dtw_example.png'
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
