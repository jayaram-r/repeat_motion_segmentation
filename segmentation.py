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
import multiprocessing
from functools import partial
import dtw
from tslearn.metrics import dtw_path as dtw_tslearn
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


def generate_sequence(n, curve='sine', noise=True, noise_level=0.005):
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
        x = np.sin(((2 * np.pi) / m) * np.arange(m))
    elif curve == 'cosine':
        x = np.cos(((2 * np.pi) / m) * np.arange(m))
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

    return x


def average_distance_to_templates(sequence, templates):
    val = 0.0
    for seq in templates:
        d, _, _, _ = dtw.accelerated_dtw(sequence[:, np.newaxis], seq[:, np.newaxis], dist='cityblock')
        val += d

    return val / len(templates)


def helper_search(sequence, templates, reverse_search, m):
    if reverse_search:
        seq = sequence[-m:]
    else:
        seq = sequence[:m]

    return m, average_distance_to_templates(seq, templates)


def search_subsequence(sequence, templates, avg_length=None, reverse_search=False):
    """
    Search for the subsequence that leads to minimum average DTW distance to the template sequences.
    The search can be from the right (reverse) by setting `reverse_search = True`.
    Specify `avg_length` to restrict the search to the range `[0.5 * avg_length, 2 * avg_length]`.

    :param sequence: numpy array of shape (N, ) with float values.
    :param templates: list of numpy arrays, where each array is a template or reference sequence.
    :param avg_length: None or an int value.
    :param reverse_search: True or False

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
        helper_search_partial = partial(helper_search, sequence, templates, reverse_search)
        pool_obj = multiprocessing.Pool(processes=num_proc)
        results = []
        _ = pool_obj.map_async(helper_search_partial, range(l_min, l_max + 1), chunksize=None,
                               callback=results.extend)
        pool_obj.close()
        pool_obj.join()
    else:
        # num_proc = 1
        results = [helper_search(sequence, templates, reverse_search, m) for m in range(l_min, l_max + 1)]

    # Return the subsequence length that leads to minimum average DTW distance to the template sequences
    return min(results, key=operator.itemgetter(1))


def segment_repeat_sequences(data, templates, num_segments):
    """

    :param data: numpy array of shape (N, ) with float values.
    :param templates: list of numpy arrays, where each array is a template or reference sequence.
    :param num_segments: integer number of segments required. Should be >= 1.

    :return:
        list of numpy arrays (of length `num_segments`) which are the segmented subsequences.
    """
    if num_segments < 1:
        raise ValueError("Invalid value '{}' for the input 'num_segments'.".format(num_segments))

    if num_segments == 1:
        logger.warning("Input 'num_segments' is 1. Returning input sequence without segmentation.")
        return [copy.deepcopy(data)]

    logger.info("Length of input sequence = %d. Number of segments required = %d", data.shape[0], num_segments)
    if num_segments == 2:
        m, d1 = search_subsequence(data, templates)
        d2 = average_distance_to_templates(data[m:], templates)
        data_segments = [data[:m], data[m:]]
        logger.info("Length of subsequence 1 = %d. Average DTW distance to the templates = %.6f", m, d1)
        logger.info("Length of subsequence 2 = %d. Average DTW distance to the templates = %.6f",
                    data.shape[0] - m, d2)
    else:
        # Starting from the left end of the sequence, find the subsequence with minimum average DTW distance from
        # the templates. Repeat this iteratively to find the segments
        data_segments = []
        data_rem = copy.copy(data)
        num_segments_rem = num_segments
        k = 1
        while num_segments_rem > 1:
            avg_length = int(np.floor(data_rem.shape[0] / float(num_segments_rem)))
            m, d1 = search_subsequence(data_rem, templates, avg_length=avg_length)
            data_segments.append(data_rem[:m])
            data_rem = data_rem[m:]
            logger.info("Length of subsequence %d = %d. Average DTW distance to the templates = %.6f", k, m, d1)
            num_segments_rem -= 1
            k += 1

        data_segments.append(data_rem)
        d1 = average_distance_to_templates(data_rem, templates)
        logger.info("Length of subsequence %d = %d. Average DTW distance to the templates = %.6f",
                    k, data_rem.shape[0], d1)

    return data_segments


def main():
    np.random.seed(seed=12345)
    # Choose one of: 'sine', 'cosine', 'gaussian', 'gaussian_inverted'
    curve = 'sine'

    # Parameters for data generation
    # Number of repetitions (segments) in the sequence
    num_repeat = 10

    # The length of each repetition subsequence is picked at random from this interval
    length_range = [100, 125]

    # Generate a set of sequences to use as templates for this action
    num_templates = 5
    template_sequences = []
    for _ in range(num_templates):
        template_sequences.append(
            generate_sequence(np.random.randint(length_range[0], high=length_range[1]), curve=curve)
        )

    # Generate and plot the data sequence
    data_sequence = []
    for _ in range(num_repeat):
        data_sequence.append(
            generate_sequence(np.random.randint(length_range[0], high=length_range[1]), curve=curve)
        )

    # Calculate the average DTW distance between each repetition subsequence and the template sequences.
    avg_dist = np.zeros(num_repeat)
    for i, seq1 in enumerate(data_sequence):
        avg_dist[i] = average_distance_to_templates(seq1, template_sequences)
        logger.info("Average DTW distance between subsequence %d and the template sequences = %.6f",
                    i + 1, avg_dist[i])

    logger.info("")
    # Concatenate the repetition subsequences into one. The segmentation algorithm takes this sequence as input.
    data_sequence_concat = np.concatenate(data_sequence)

    # Perform segmentation of the concateneted sequence
    data_segments = segment_repeat_sequences(data_sequence_concat, template_sequences, num_repeat)

    # Plot the concatenated sequence and the result of the segmentation
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(data_sequence_concat.shape[0]), data_sequence_concat, linestyle='--', color='r',
             marker='.', markersize=4)
    # ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')
    ax1.set_title('Segmentation using DTW matching', fontsize=10, fontweight='bold')

    ax1 = fig.add_subplot(2, 1, 2)
    st = 0
    nc = len(COLORS_LIST)
    for j, seg in enumerate(data_segments):
        en = st + seg.shape[0] - 1
        ax1.plot(np.arange(st, en + 1), seg, linestyle='--', color=COLORS_LIST[j % nc], marker='.', markersize=4)
        st = en + 1

    ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')
    plt.plot()
    plot_file = 'segmentation.png'
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
