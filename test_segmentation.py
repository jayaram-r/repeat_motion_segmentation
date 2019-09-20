"""
Segmenting a time series with multiple repetitions of a given action.
"""
import numpy as np
import os
import time
import random
import pickle
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
    preprocess_templates,
    template_info_tuple
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# https://stackoverflow.com/a/37232760
COLORS_LIST = ['grey', 'r', 'b', 'g', 'c', 'orange', 'm', 'lawngreen', 'gold', 'y', 'hotpink', 'blueviolet',
               'crimson', 'lavender', 'khaki', 'darkseagreen', 'darkturquoise', 'brown', 'firebrick', 'maroon',
               'orchid', 'teal', 'fuchsia', 'indigo', 'palevioletred', 'rosybrown']


def generate_test_data():
    np.random.seed(seed=1234)
    # Choose one of: 'sine', 'cosine', 'gaussian', 'gaussian_inverted'
    curve = 'sine'

    # The length of each repetition subsequence is picked at random from this interval
    length_range = [100, 125]
    length_range_noise = [20, 30]

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
                m = np.random.randint(length_range_noise[0], high=length_range_noise[1])
                data_sequence.append(0.01 * np.random.rand(m, 1))

    # Randomize the order of the sequences
    random.shuffle(data_sequence)

    # Concatenate the repetition subsequences into one. The segmentation algorithm takes this sequence as input.
    data_sequence = np.concatenate(data_sequence)

    return template_sequences, data_sequence


def segment_and_plot_results(template_sequences, data_sequence, output_direc):
    # Value between 0 and 1 specifying the width of the Sakoe-Chiba window in terms of the length of the
    # longer sequence
    warping_window = 0.25
    # Value between 0 and 1 that controls the search range of the subsequence length
    alpha = 0.75

    # Create the output directory if required
    if not os.path.isdir(output_direc):
        os.makedirs(output_direc)

    # Preprocess the template sequences and calculate the upper threshold on the average DTW distance corresponding
    # to each action.
    # Results of the template preprocessing are saved in a Pickle file that can be loaded and reused directly
    # on subsequent runs
    results_file = os.path.join(output_direc, 'template_results.pkl')
    results = None
    if os.path.isfile(results_file):
        with open(results_file, 'rb') as fp:
            results = pickle.load(fp)

    if results is None:
        t1 = time.time()
        results = preprocess_templates(
            template_sequences, normalize=True, normalization_type='z-score', warping_window=warping_window,
            alpha=alpha, templates_results_file=results_file
        )
        t2 = time.time()
        logger.info("Time taken for preprocessing the templates = %.2f seconds", t2 - t1)
    else:
        logger.info("Loading preprocessed template results from the file: %s", results_file)

    logger.info("")
    t1 = time.time()
    # Perform segmentation of the data sequence
    data_segments, labels = segment_repeat_sequences(
        data_sequence, results['templates_normalized'], results['templates_info'], results['distance_thresholds'],
        results['length_stats'], normalize=True, normalization_type='z-score', warping_window=warping_window
    )
    t2 = time.time()
    logger.info("Time taken for segmentation = %.2f seconds", t2 - t1)
    logger.info("")

    # Plot the original sequence
    dim = data_sequence.shape[1]
    fig = plt.figure()
    index_vals = np.arange(data_sequence.shape[0])
    for j in range(dim):
        ax1 = fig.add_subplot(dim, 1, j + 1)
        ax1.plot(index_vals, data_sequence[:, j], linestyle='--', color='r', marker='.', markersize=4)
        if j == (dim - 1):
            ax1.set_xlabel("time index (t)", fontsize=10, fontweight='bold')

        ax1.set_ylabel(r"$x_{}[t]$".format(j + 1), fontsize=10, fontweight='bold', rotation=0)
        if j == 0:
            ax1.set_title('Input sequence', fontsize=10, fontweight='bold')

    plt.plot()
    plot_file = os.path.join(output_direc, 'sequence_plot.png')
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')

    # Plot the segmented sequence
    nc = len(COLORS_LIST)
    fig = plt.figure()
    for j in range(dim):
        ax1 = fig.add_subplot(dim, 1, j + 1)
        st = 0
        for lab, seg in zip(labels, data_segments):
            en = st + seg.shape[0]
            ax1.plot(np.arange(st, en), seg[:, j], linestyle='--', color=COLORS_LIST[lab % nc],
                     marker='.', markersize=4)
            st = en

        if j == (dim - 1):
            ax1.set_xlabel("time index (t)", fontsize=10, fontweight='bold')

        ax1.set_ylabel(r"$x_{}[t]$".format(j + 1), fontsize=10, fontweight='bold', rotation=0)
        if j == 0:
            ax1.set_title('Sequence segmentation result', fontsize=10, fontweight='bold')

    plt.plot()
    plot_file = os.path.join(output_direc, 'sequence_segmented_plot.png')
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')


def main():
    # Generate synthetic template data and the test data sequence
    template_sequences, data_sequence = generate_test_data()

    # Perform segmentation of the data sequence and plot the results
    output_direc = os.path.join(os.getcwd(), 'results')
    segment_and_plot_results(template_sequences, data_sequence, output_direc)


if __name__ == '__main__':
    main()
