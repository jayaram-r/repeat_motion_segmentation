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
import numpy as np
import copy
import multiprocessing
from functools import partial
from scipy import stats
import dtw
import logging
from repeat_motion_segmentation.utils import normalize_maxmin


METRIC = 'cityblock'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def average_distance_to_templates(sequence, templates):
    val_min = np.inf
    label = 1
    for i, t in enumerate(templates, start=1):
        val = 0.0
        for seq in t:
            d, _, _, _ = dtw.accelerated_dtw(sequence, seq, dist=METRIC)
            val += d

        val /= len(t)
        if val < val_min:
            val_min = val
            label = i

    return val_min, label


def helper_dtw_distance(sequence, templates, index_tuple):
    d, _, _, _ = dtw.accelerated_dtw(sequence, templates[index_tuple[0]][index_tuple[1]], dist=METRIC)
    return index_tuple[0], index_tuple[1], d


def average_distance_to_templates_parallel(sequence, templates, num_proc):
    num_actions = len(templates)
    indices = [(i, j) for i in range(num_actions) for j in range(len(templates[i]))]
    helper_partial = partial(helper_dtw_distance, sequence, templates)
    pool_obj = multiprocessing.Pool(processes=num_proc)
    results = []
    _ = pool_obj.map_async(helper_partial, indices, chunksize=None, callback=results.extend)
    pool_obj.close()
    pool_obj.join()

    avg_dist = np.zeros(num_actions)
    cnt = np.zeros(num_actions)
    for tup in results:
        avg_dist[tup[0]] += tup[2]
        cnt[tup[0]] += 1.0

    avg_dist = avg_dist / np.clip(cnt, 1e-6, None)
    label = np.argmin(avg_dist)

    return avg_dist[label], label + 1


def normalize_subsequence(sequence, m, sequence_mean, sequence_stdev, sequence_min, sequence_max,
                          normalize, normalization_type):
    if normalize:
        if normalization_type == 'z-score':
            sequence_norm = (sequence[:m, :] - sequence_mean[m - 1]) / sequence_stdev[m - 1]
        else:
            # Max-min normalization
            if sequence_max[m - 1] > sequence_min[m - 1]:
                sequence_norm = (sequence[:m, :] - sequence_min[m - 1]) / (sequence_max[m - 1] - sequence_min[m - 1])
            else:
                # Maximum and minimum values are equal. Setting all normalized values to 1
                sequence_norm = np.ones_like(sequence[:m, :])

    else:
        sequence_norm = sequence[:m, :]

    return sequence_norm


def search_subsequence(sequence, templates, min_length, max_length, normalize=True, normalization_type='z-score'):
    """
    Search for the subsequence that leads to minimum average DTW distance to the template sequences.

    :param sequence: numpy array of shape (N, 1) with float values.
    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, 1)) corresponding to a template sequence.
    :param min_length: minimum length of the subsequence.
    :param max_length: maximum length of the subsequence.
    :param normalize: Apply normalization to the templates and the data subsequences, if set to True.
    :param normalization_type: Type of normalization to apply. Should be set to 'z-score' or 'max-min'.

    :return: tuple (a, b, c), where `a` is the best subsequence length, `b` is the minimum average DTW distance,
             and `c` is the label of the best-matching template.
    """
    N = sequence.shape[0]
    max_length = min(N, max_length)
    num_proc = max(1, multiprocessing.cpu_count() - 1)
    if num_proc > 1:
        parallel = True
    else:
        parallel = False

    sequence_min = None
    sequence_max = None
    sequence_mean = None
    sequence_stdev = None
    if normalize:
        if normalization_type == 'z-score':
            # Calculate the rolling mean and standard-deviation of the entire data sequence. This will be used
            # for z-score normalization of subsequences of different lengths
            den = np.arange(1, N + 1).astype(np.float)
            sequence_mean = np.cumsum(sequence) / den
            arr = sequence[:, 0] - sequence_mean
            sequence_stdev = np.sqrt(np.clip(np.cumsum(arr * arr) / den, 1e-16, None))
        else:
            # Calculate the rolling maximum and minimum of the entire data sequence. This will be used in
            # the max-min normalization of subsequences of different lengths
            sequence_min = np.minimum.accumulate(sequence[:, 0])
            sequence_max = np.maximum.accumulate(sequence[:, 0])

    # Calculate the minimum average DTW distance to the templates for the average subsequence length
    # (between `min_length` and `max_length`). This gives a good reference value for distance-based pruning.
    len_best = int(np.round(0.5 * (min_length + max_length)))
    sequence_norm = normalize_subsequence(sequence, len_best, sequence_mean, sequence_stdev, sequence_min,
                                          sequence_max, normalize, normalization_type)
    if parallel:
        d_min, label_best = average_distance_to_templates_parallel(sequence_norm, templates, num_proc)
    else:
        d_min, label_best = average_distance_to_templates(sequence_norm, templates)

    search_set = set(range(min_length, max_length + 1)) - {len_best}
    for m in search_set:
        sequence_norm = normalize_subsequence(sequence, m, sequence_mean, sequence_stdev, sequence_min,
                                              sequence_max, normalize, normalization_type)
        if parallel:
            d, label = average_distance_to_templates_parallel(sequence_norm, templates, num_proc)
        else:
            d, label = average_distance_to_templates(sequence_norm, templates)

        if d < d_min:
            d_min = d
            label_best = label
            len_best = m

    return len_best, d_min, label_best


def segment_repeat_sequences(data, templates, normalize=True, normalization_type='z-score'):
    """
    Segment the sequence `data` to closely match the sequences specified in the list `templates`.

    :param data: numpy array of shape (N, 1) with float values corresponding to the data sequence.
    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, 1)) corresponding to a template sequence.
    :param normalize: Apply normalization to the templates and the data subsequences, if set to True.
    :param normalization_type: Type of normalization to apply. Should be set to 'z-score' or 'max-min'.

    :return:
        data_segments: list of segmented subsequences, each of which are numpy arrays of shape (m, 1).
        labels: list of best-matching template labels for the subsequences, where value `i` corresponds to the
                templates in position `i - 1` of the input list `templates`.
    """
    if normalization_type not in ('z-score', 'max-min'):
        raise ValueError("Invalid value '{}' for the parameter 'normalization_type'.".format(normalization_type))

    # Use the length of the template sequences as reference to define the search range for the subsequences
    l_min, l_avg, l_max = np.percentile([b.shape[0] for a in templates for b in a], [0, 50, 100])
    min_length = int(min(l_min, max(2, np.floor(0.5 * l_avg))))
    max_length = int(max(l_max, np.ceil(2.0 * l_avg)))

    logger.info("Length of the input sequence = %d.", data.shape[0])
    num_actions = len(templates)
    logger.info("Number of actions to be matched = %d.", num_actions)
    if normalize:
        logger.info("Applying '%s' normalization to the template sequences.", normalization_type)
        templates_norm = [[]] * num_actions
        for i in range(num_actions):
            num_templates = len(templates[i])
            templates_norm[i] = [[]] * num_templates
            for j in range(num_templates):
                if normalization_type == 'z-score':
                    templates_norm[i][j] = stats.zscore(templates[i][j])
                else:
                    templates_norm[i][j] = normalize_maxmin(templates[i][j])

    else:
        templates_norm = templates

    # Starting from the left end of the sequence, find the subsequence with minimum average DTW distance from
    # the templates. Repeat this iteratively to find the segments
    logger.info("Search range for the subsequence length = [%d, %d].", min_length, max_length)
    data_segments = []
    labels = []
    data_rem = copy.copy(data)
    k = 1
    while data_rem.shape[0] > min_length:
        m, d_min, label = search_subsequence(data_rem, templates_norm, min_length, max_length,
                                             normalize=normalize, normalization_type=normalization_type)
        data_segments.append(data_rem[:m, :])
        labels.append(label)
        data_rem = data_rem[m:, :]
        logger.info("Length of subsequence %d = %d. Matched template label = %d. Average DTW distance to the "
                    "matched templates = %.6f.", k, m, label, d_min)
        k += 1

    return data_segments, labels
