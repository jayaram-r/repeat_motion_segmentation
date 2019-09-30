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
    template_labels = []
    for i, tp in enumerate([1.0, 0.5, 0.25, 2.0], start=1):
        a = []
        for _ in range(num_templates):
            a.append(
                generate_sequence(np.random.randint(length_range[0], high=length_range[1]), curve=curve, tp=tp)
            )

        template_sequences.append(a)
        # Second element of the tuple is just a dummy value of 1. It is used in the motion sensor data to
        # denote the category of speed: "slow", "normal", or "fast"
        template_labels.append([(i, 'normal') for _ in a])

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

    return template_sequences, template_labels, data_sequence


def plot_templates(templates, template_labels, output_direc):
    if not os.path.isdir(output_direc):
        os.makedirs(output_direc)

    for i in range(len(templates)):
        direc = os.path.join(output_direc, template_labels[i][0][0])
        if not os.path.isdir(direc):
            os.makedirs(direc)

        for j in range(len(templates[i])):
            seq = templates[i][j]
            lab = template_labels[i][j]
            lab_str = "{}, {}".format(lab[0], lab[1])
            dim = seq.shape[1]

            y_min = np.min(seq)
            # Slightly smaller value
            if y_min > 0:
                y_min = 0.99 * y_min
            else:
                y_min = 1.01 * y_min

            y_max = np.max(seq)
            # Slightly larger value
            if y_max > 0:
                y_max = 1.01 * y_max
            else:
                y_max = 0.99 * y_max

            fig = plt.figure()
            index_vals = np.arange(seq.shape[0])
            for k in range(dim):
                ax1 = fig.add_subplot(dim, 1, k + 1)
                ax1.plot(index_vals, seq[:, k], linestyle='--', color='g', marker='.', markersize=4)
                if k == (dim - 1):
                    ax1.set_xlabel("time index (t)", fontsize=10, fontweight='bold')

                ax1.set_ylabel(r"$x_{}[t]$".format(k + 1), fontsize=10, fontweight='bold', rotation=0)
                ax1.set_ylim(y_min, y_max)
                if k == 0:
                    ax1.set_title(lab_str, fontsize=10, fontweight='bold')

            plt.plot()
            plot_file = os.path.join(direc, '{}_{}_{:d}.png'.format(lab[0], lab[1], j + 1))
            fig.savefig(plot_file, dpi=600, bbox_inches='tight')
            plt.close()


def segment_and_plot_results(template_sequences, template_labels, data_sequence, output_direc,
                             features_per_action=None,
                             warping_window=0.25,
                             normalize=True,
                             num_templates_to_select=5,
                             length_step=1,
                             offset_step=1,
                             max_overlap=10,
                             min_length=2,
                             max_length=1000000,
                             approx=False):
    """

    :param template_sequences: list of template sequences corresponding to each action.
    :param template_labels: list of template labels corresponding to each action.
    :param data_sequence: numpy array of shape `(N, d)` specifying the data sequence to be segmented. N is the
                          length and d is the dimension of the sequence.
    :param output_direc: output directory name.
    :param features_per_action: None or a dict mapping each action label to a list of feature indices
                               (starting at 0) that are to be used for that action. If None, all the features will
                               be used for all the actions.
    :param warping_window: Value in (0, 1] specifying the size of the Sakoe-Chiba band used for constraining the DTW
                           paths.
    :param normalize: Set to True in order to apply z-score normalization.
    :param num_templates_to_select: Number of templates to select for segmentation and matching.
    :param length_step: (int) length search is done in increments of this step. Default value is 1.
    :param offset_step: (int) offset search is done in increments of this step. Default value is 1.
    :param max_overlap: (int) maximum allowed overlap between successive segments. Set to 0 for no overlap.
    :param min_length: A minimum length can be specified to limit the search range.
    :param max_length: A maximum length can be specified to limit the search range.
    :param approx: set to True to enable a coarse but faster search over the offsets.

    :return: None
    """
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
            template_sequences, template_labels, features_per_action=features_per_action, normalize=normalize,
            warping_window=warping_window, num_templates_to_select=num_templates_to_select,
            templates_results_file=results_file, min_length=min_length, max_length=max_length
        )
        t2 = time.time()
        logger.info("Time taken for preprocessing the templates = %.2f seconds", t2 - t1)
    else:
        logger.info("Loading preprocessed template results from the file: %s", results_file)

    logger.info("")
    t1 = time.time()
    # Perform segmentation of the data sequence
    template_labels = results['template_labels']
    data_segments, labels = segment_repeat_sequences(
        data_sequence, results['templates_normalized'], results['templates_info'], results['template_counts'],
        results['feature_mask_per_action'], results['distance_thresholds'], results['length_stats'],
        normalize=normalize, warping_window=warping_window, length_step=length_step, offset_step=offset_step,
        max_overlap=max_overlap, approx=approx
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
    check_legend = set()
    fig = plt.figure()
    for j in range(dim):
        ax1 = fig.add_subplot(dim, 1, j + 1)
        st = 0
        nseg = 0
        for lab, seg in zip(labels, data_segments):
            if max_overlap > 0 and nseg > 0 and lab == 0:
                # First few points will be overlapping with the previous segment and should not be plotted again
                seg = seg[max_overlap:, :]

            # Adding legend only for the first subplot pane
            if j == 0:
                lab_str = str(template_labels[lab - 1][0][0]) if lab > 0 else 'No match'
                if lab_str in check_legend:
                    lab_str = None
                else:
                    check_legend.add(lab_str)
            else:
                lab_str = None

            en = st + seg.shape[0]
            if lab_str:
                ax1.plot(np.arange(st, en), seg[:, j], linestyle='--', color=COLORS_LIST[lab % nc],
                         marker='.', markersize=4, label=lab_str)
            else:
                ax1.plot(np.arange(st, en), seg[:, j], linestyle='--', color=COLORS_LIST[lab % nc],
                         marker='.', markersize=4)
            st = en
            nseg += 1

        if j == (dim - 1):
            ax1.set_xlabel("time index (t)", fontsize=10, fontweight='bold')

        ax1.set_ylabel(r"$x_{}[t]$".format(j + 1), fontsize=10, fontweight='bold', rotation=0)
        if j == 0:
            ax1.set_title('Sequence segmentation result', fontsize=10, fontweight='bold')
            ax1.legend(loc='best', fontsize=8)

    plt.plot()
    plot_file = os.path.join(output_direc, 'sequence_segmented_plot.png')
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')


def main():
    # Generate synthetic template data and the test data sequence
    template_sequences, template_labels, data_sequence = generate_test_data()

    # Perform segmentation of the data sequence and plot the results
    output_direc = os.path.join(os.getcwd(), 'results')
    segment_and_plot_results(template_sequences, template_labels, data_sequence, output_direc)


if __name__ == '__main__':
    main()
