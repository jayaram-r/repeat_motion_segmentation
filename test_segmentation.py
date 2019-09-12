"""
Segmenting a time series with multiple repetitions of a given action.
"""
import numpy as np
import time
import random
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from repeat_motion_segmentation.utils import (
    gaussian_sequence,
    generate_sequence
)
from repeat_motion_segmentation.segmentation import (
    segment_repeat_sequences,
    preprocess_templates
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# https://stackoverflow.com/a/37232760
COLORS_LIST = ['grey', 'r', 'b', 'g', 'c', 'orange', 'm', 'lawngreen', 'gold', 'y', 'hotpink', 'blueviolet']


def main():
    np.random.seed(seed=123)
    # Choose one of: 'sine', 'cosine', 'gaussian', 'gaussian_inverted'
    curve = 'sine'

    # The length of each repetition subsequence is picked at random from this interval
    length_range = [100, 125]
    length_noise = [15, 25]

    # Generate a set of sequences to use as templates for this action
    num_templates = 10
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
    for tp, num_repeat in [(0.25, 3), (0.5, 3), (1.0, 3), (2.0, 3)]:
        for _ in range(num_repeat):
            data_sequence.append(
                generate_sequence(np.random.randint(length_range[0], high=length_range[1]), curve=curve, tp=tp)
            )
            if np.random.rand() < 0.5:
                # Append a noise sequence
                m = np.random.randint(length_noise[0], high=length_noise[1])
                data_sequence.append(0.01 * np.random.rand(m, 1))

    # Randomize the order of the sequences
    random.shuffle(data_sequence)

    # Concatenate the repetition subsequences into one. The segmentation algorithm takes this sequence as input.
    data_sequence = np.concatenate(data_sequence)
    warping_window = 0.25
    alpha = 0.75

    # Preprocess the template sequences and calculate the upper threshold on the average DTW distance corresponding
    # to each action
    t1 = time.time()
    templates_norm, templates_info, distance_thresholds, search_range = preprocess_templates(
        template_sequences, normalize=True, normalization_type='z-score', warping_window=warping_window, alpha=alpha
    )
    t2 = time.time()
    logger.info("Time taken for preprocessing the templates = %.2f seconds", t2 - t1)
    logger.info("")

    t1 = time.time()
    # Perform segmentation of the data sequence
    data_segments, labels = segment_repeat_sequences(
        data_sequence, templates_norm, templates_info, distance_thresholds, search_range, normalize=True,
        normalization_type='z-score', warping_window=warping_window
    )
    t2 = time.time()
    logger.info("Time taken for segmentation = %.2f seconds", t2 - t1)
    logger.info("")

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
    for lab, seg in zip(labels, data_segments):
        en = st + seg.shape[0] - 1
        ax1.plot(np.arange(st, en + 1), seg[:, 0], linestyle='--', color=COLORS_LIST[lab % nc],
                 marker='.', markersize=4)
        st = en + 1

    ax1.set_xlabel("index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("value", fontsize=10, fontweight='bold')
    plt.plot()
    plot_file = 'segmentation.png'
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    main()
