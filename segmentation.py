"""
Segmenting a time series with multiple repetitions of a given action.

DTW implentations source:
https://github.com/pierre-rouanet/dtw

tslearn package:
https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.metrics.html

We can also use FastDTW, which has linear time and space complexity in the length of the longer sequence.
An open source implementation of FastDTW can be found here:
https://pypi.org/project/fastdtw/

"""
import os
import numpy as np
import copy
import operator
import time
import random
import multiprocessing
from functools import partial
import dtw
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# https://stackoverflow.com/a/37232760
COLORS_LIST = ['r', 'b', 'g', 'c', 'orange', 'm', 'lawngreen', 'gold', 'grey', 'y', 'hotpink', 'blueviolet']


def gaussian_sequence(n, inverted=False):
    mu = (n - 1) / 2.0
    sig = (n - mu) / 2.5
    x = np.arange(n)
    t = (0.5 / (sig * sig)) * ((x - mu) ** 2)
    y = np.exp(-t)
    if inverted:
        y = 1.0 - y

    return y


def generate_sequence(n, curve='sine', noise=True, noise_level=0.005, tp=1.0):
    if n < 1:
        logger.warning("Sequence length is 0. Returning empty array")
        return np.array([])

    ni = nf = 0
    if noise:
        a = (0.15 - 0.1) * np.random.rand() + 0.1
        ni = max(1, int(np.floor(a * n)))
        a = (0.15 - 0.1) * np.random.rand() + 0.1
        nf = max(1, int(np.floor(a * n)))

    m = n - ni - nf
    if curve == 'sine':
        x = np.sin(((2.0 * np.pi) / (tp * m)) * np.arange(m))
    elif curve == 'cosine':
        x = np.cos(((2.0 * np.pi) / (tp * m)) * np.arange(m))
    elif curve == 'gaussian':
        x = gaussian_sequence(m)
    elif curve == 'gaussian_inverted':
        x = gaussian_sequence(m, inverted=True)
    else:
        raise ValueError("Invalid value '{}' for parameter 'curve'".format(curve))

    if noise:
        x = x + noise_level * np.random.rand(x.shape[0])
        # Append noise values close to 0 of length `noise_interval` at the start and end of the sequence
        if ni > 0:
            xi = noise_level * np.random.rand(ni)
            x = np.concatenate([xi, x])

        if nf > 0:
            xf = noise_level * np.random.rand(nf)
            x = np.concatenate([x, xf])

    return x[:, np.newaxis]


def average_distance_to_templates(sequence, templates):
    val_min = np.inf
    label = 1
    for i, t in enumerate(templates, start=1):
        val = 0.0
        for seq in t:
            d, _, _, _ = dtw.accelerated_dtw(sequence, seq, dist='cityblock')
            val += d

        val /= len(t)
        if val < val_min:
            val_min = val
            label = i

    return val_min, label


def helper_search(sequence, templates, m):
    ret = average_distance_to_templates(sequence[:m, :], templates)
    return m, ret[0], ret[1]


def search_subsequence(sequence, templates, avg_length=None):
    """
    Search for the subsequence that leads to minimum average DTW distance to the template sequences.

    :param sequence: numpy array of shape (N, ) with float values.
    :param templates: list of numpy arrays, where each array is a template or reference sequence.
    :param avg_length: None or an int value.

    :return: tuple (a, b), where `a` is the best subsequence length and `b` is the minimum average distance.
    """
    N = sequence.shape[0]
    if avg_length is None:
        l_min = 2
        l_max = N - 1
    else:
        l_min = max(2, int(np.floor(0.5 * avg_length)))
        l_max = min(N - 1, int(np.ceil(2.0 * avg_length)))

    num_proc = max(1, multiprocessing.cpu_count() - 1)
    if num_proc > 1:
        helper_search_partial = partial(helper_search, sequence, templates)
        pool_obj = multiprocessing.Pool(processes=num_proc)
        results = []
        _ = pool_obj.map_async(helper_search_partial, range(l_min, l_max + 1), chunksize=None,
                               callback=results.extend)
        pool_obj.close()
        pool_obj.join()
    else:
        # num_proc = 1
        results = [helper_search(sequence, templates, m) for m in range(l_min, l_max + 1)]

    # Return the subsequence length that leads to minimum average DTW distance to the template sequences
    return min(results, key=operator.itemgetter(1))


def segment_repeat_sequences(data, templates):
    """
    Segment the sequence `data` to closely match the sequences specified in the list `templates`.

    :param data: numpy array of shape (N, 1) with float values corresponding to the data sequence.
    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, 1)) corresponding to a template sequence.
    :return:
        list of segmented subsequences, each of which are numpy arrays of shape (m, 1).
    """
    # Average length of templates and minimum length required to be considered for segmentation
    avg_length = int(np.round(np.mean([b.shape[0] for a in templates for b in a])))
    min_length = max(2, int(np.floor(0.5 * avg_length)))

    # Starting from the left end of the sequence, find the subsequence with minimum average DTW distance from
    # the templates. Repeat this iteratively to find the segments
    data_segments = []
    data_rem = copy.copy(data)
    k = 1
    while data_rem.shape[0] >= min_length:
        m, d_min, label = search_subsequence(data_rem, templates, avg_length=avg_length)
        data_segments.append(data_rem[:m, :])
        data_rem = data_rem[m:, :]
        logger.info("Length of subsequence %d = %d. Matched template label = %d. Average DTW distance to the "
                    "matched templates = %.6f.", k, m, label, d_min)
        k += 1

    return data_segments


def main():
    np.random.seed(seed=183)
    # Choose one of: 'sine', 'cosine', 'gaussian', 'gaussian_inverted'
    curve = 'sine'

    # The length of each repetition subsequence is picked at random from this interval
    length_range = [100, 125]

    # Generate a set of sequences to use as templates for this action
    num_templates = 5
    template_sequences = []
    for tp in [1.0, 0.5, 0.25, 2.0]:
        a = []
        for _ in range(num_templates):
            a.append(
                generate_sequence(np.random.randint(length_range[0], high=length_range[1]), curve=curve, tp=tp)
            )

        template_sequences.append(a)

    # Generate the data sequence
    data_sequence = []
    for tp, num_repeat in [(0.25, 4), (0.5, 3), (1.0, 5), (2.0, 3)]:
        for _ in range(num_repeat):
            data_sequence.append(
                generate_sequence(np.random.randint(length_range[0], high=length_range[1]), curve=curve, tp=tp)
            )

    # Randomize the order of the sequences
    random.shuffle(data_sequence)

    # Concatenate the repetition subsequences into one. The segmentation algorithm takes this sequence as input.
    data_sequence = np.concatenate(data_sequence)

    t1 = time.time()
    # Perform segmentation of the concateneted sequence
    data_segments = segment_repeat_sequences(data_sequence, template_sequences)
    t2 = time.time()
    logger.info("Time taken for segmentation = %.2f seconds", t2 - t1)

    # Plot the concatenated sequence and the result of the segmentation
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(data_sequence.shape[0]), data_sequence[:, 0], linestyle='--', color='r',
             marker='.', markersize=4)
    # ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')
    ax1.set_title('Segmentation using DTW matching', fontsize=10, fontweight='bold')

    ax1 = fig.add_subplot(2, 1, 2)
    st = 0
    nc = len(COLORS_LIST)
    for j, seg in enumerate(data_segments):
        en = st + seg.shape[0] - 1
        ax1.plot(np.arange(st, en + 1), seg[:, 0], linestyle='--', color=COLORS_LIST[j % nc],
                 marker='.', markersize=4)
        st = en + 1

    ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')
    plt.plot()
    plot_file = 'segmentation.png'
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
