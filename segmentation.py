"""
Segmenting a time series with multiple repetitions of a given action.

Dynamic time warping (DTW) implementation by the DTAI research group, which has fast implementations (with C
bindings), is used. Documentation and installation instructions can be found below:
https://github.com/wannesm/dtaidistance
https://dtaidistance.readthedocs.io/en/latest/index.html

"""
import numpy as np
import copy
from collections import namedtuple
import multiprocessing
from functools import partial
from scipy import stats
from dtaidistance import dtw
import logging
from repeat_motion_segmentation.utils import normalize_maxmin


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def average_lb_distance_to_templates1(sequence, templates_info):
    """
    Simple but fast lower bound to the DTW based on comparison of only the first and last values of the sequence
    and the templates.
    """
    len_seq = sequence.shape[0]
    val_min = np.inf
    for t in templates_info:
        val = 0.0
        for temp in t:
            val += (np.sqrt((sequence[0, 0] - temp.first_value) ** 2 + (sequence[-1, 0] - temp.last_value) ** 2) /
                    (len_seq + temp.length))

        val /= len(t)
        if val < val_min:
            val_min = val

    return val_min


def average_lb_distance_to_templates2(sequence, templates, templates_info):
    """
    Minimum average distance to the templates, where the distance is the lower bound of the DTW distance proposed by:
    Yi, Byoung-Kee, H. V. Jagadish, and Christos Faloutsos. "Efficient retrieval of similar time sequences
    under time warping." Proceedings 14th International Conference on Data Engineering. IEEE, 1998.

    Minor modifications are made to the method proposed by Yi et. al. The exact squared difference between the first
    and the last values are computed, since that is a requirement for DTW distance. The minimum-maximum envelope
    of the template sequence is used to calculate the first lower bound, and the second lower bound is calculated
    by using the minimum-maximum envelope of the input sequence. The maximum of both these values is taken as the
    final lower bound.
    """
    len_seq = sequence.shape[0]
    sequence = sequence[:, 0]
    min_seq = np.min(sequence)
    max_seq = np.max(sequence)

    val_min = np.inf
    for i, t in enumerate(templates):
        val = 0.0
        for j, temp in enumerate(t):
            info = templates_info[i][j]
            # For the first and last values of the sequence, we calculate the exact deviation instead of
            # deviation with the maximum or minimum
            dev_first_last = (sequence[0] - info.first_value) ** 2 + (sequence[-1] - info.last_value) ** 2

            # First lower bound calculated using the maximum and minimum values of the template as the envelope
            mask1 = sequence > info.max
            mask2 = sequence < info.min
            mask1[0] = mask1[-1] = mask2[0] = mask2[-1] = False
            val1 = (np.sqrt(np.sum((sequence[mask1] - info.max) ** 2) +
                            np.sum((sequence[mask2] - info.min) ** 2) + dev_first_last) /
                    (len_seq + info.length))

            # Second lower bound calculated using the maximum and minimum values of the sequence as the envelope
            mask1 = temp[:, 0] > max_seq
            mask2 = temp[:, 0] < min_seq
            mask1[0] = mask1[-1] = mask2[0] = mask2[-1] = False
            val2 = (np.sqrt(np.sum((temp[mask1, 0] - max_seq) ** 2) +
                            np.sum((temp[mask2, 0] - min_seq) ** 2) + dev_first_last) /
                    (len_seq + info.length))

            # Maximum of the two lower bounds is still a lower bound. This is added up to compute the average
            val += max(val1, val2)

        val /= len(t)
        if val < val_min:
            val_min = val

    return val_min


def average_distance_to_templates(sequence, templates, warping_window):
    len_seq = sequence.shape[0]
    val_min = np.inf
    label = 1
    for i, t in enumerate(templates, start=1):
        val = 0.0
        for temp in t:
            len_temp = temp.shape[0]
            if warping_window is not None:
                warping_window = int(np.ceil(warping_window * max(len_seq, len_temp)))
                # If the sequences have different lengths, the warping window cannot be smaller than the difference
                # between the length of the sequences
                warping_window = max(warping_window, abs(len_seq - len_temp + 1))

            d = (dtw.distance_fast(sequence[:, 0], temp[:, 0], window=warping_window)) / float(len_seq + len_temp)
            val += d

        val /= len(t)
        if val < val_min:
            val_min = val
            label = i

    return val_min, label


def helper_dtw_distance(sequence, templates, warping_window, index_tuple):
    t = templates[index_tuple[0]][index_tuple[1]]
    len_seq = sequence.shape[0]
    len_temp = t.shape[0]
    if warping_window is not None:
        warping_window = int(np.ceil(warping_window * max(len_seq, len_temp)))
        # If the sequences have different lengths, the warping window cannot be smaller than the difference
        # between the length of the sequences
        warping_window = max(warping_window, abs(len_seq - len_temp + 1))

    d = (dtw.distance_fast(sequence[:, 0], t[:, 0], window=warping_window)) / float(len_seq + len_temp)

    return index_tuple[0], index_tuple[1], d


def average_distance_to_templates_parallel(sequence, templates, warping_window, num_proc):
    num_actions = len(templates)
    indices = [(i, j) for i in range(num_actions) for j in range(len(templates[i]))]
    helper_partial = partial(helper_dtw_distance, sequence, templates, warping_window)
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


def search_subsequence(sequence, templates, templates_info, min_length, max_length, normalize=True,
                       normalization_type='z-score', warping_window=None, use_lower_bounds=True):
    """
    Search for the subsequence that leads to minimum average DTW distance to the template sequences.

    :param sequence: numpy array of shape (N, 1) with float values.
    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, 1)) corresponding to a template sequence.
    :param templates_info: list similar to `templates`, but each element of the list is a namedtuple with information
                           about the template such as length, minimum value, and maximum value.
    :param min_length: minimum length of the subsequence.
    :param max_length: maximum length of the subsequence.
    :param normalize: Apply normalization to the templates and the data subsequences, if set to True.
    :param normalization_type: Type of normalization to apply. Should be set to 'z-score' or 'max-min'.
    :param warping_window: Size of the warping window used to constrain the DTW matching path. This is also know as
                           the Sakoe-Chiba band in DTW literature. This can be set to `None` if no warping window
                           constraint is to be applied; else it should be set to a fractional value in (0, 1]. The
                           actual warping window is obtained by multiplying this fraction with the length of the
                           longer sequence. Suppose this window value is `w`, then any point `(i, j)` along the DTW
                           path satisfies `|i - j| <= w`. Setting this to a large value (closer to 1), allows the
                           warping path to be flexible, while setting it to a small value (closer to 0) will constrain
                           the warping path to be closer to the diagonal. Note that a small value can also speed-up
                           the DTW calculation significantly.
    :param use_lower_bounds: Set to `True` to use the lower bounds of the DTW. This will speed up the search.

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
        d_min, label_best = average_distance_to_templates_parallel(sequence_norm, templates, warping_window,
                                                                   num_proc)
    else:
        d_min, label_best = average_distance_to_templates(sequence_norm, templates, warping_window)

    search_set = set(range(min_length, max_length + 1)) - {len_best}
    search_set = np.random.permutation(list(search_set))
    for m in search_set:
        sequence_norm = normalize_subsequence(sequence, m, sequence_mean, sequence_stdev, sequence_min,
                                              sequence_max, normalize, normalization_type)
        if use_lower_bounds:
            # Cascading lower bound distances to the DTW for fast pruning of bad (non-match) sequences.
            # Lower bound 1 based on comparison of only the first and last values of the sequence and template.
            d_lb1 = average_lb_distance_to_templates1(sequence_norm, templates_info)
            # If the minimum average distance to the templates based on this lower bound is larger than the current
            # minimum `d_min`, then there is no need to calculate the DTW distances to the templates
            if d_lb1 > d_min:
                continue

            # Lower bound 2 based on comparison with a precomputed lower and upper bound to the template sequences.
            d_lb2 = average_lb_distance_to_templates2(sequence_norm, templates, templates_info)
            if d_lb2 > d_min:
                continue

        if parallel:
            d, label = average_distance_to_templates_parallel(sequence_norm, templates, warping_window, num_proc)
        else:
            d, label = average_distance_to_templates(sequence_norm, templates, warping_window)

        if d < d_min:
            d_min = d
            label_best = label
            len_best = m

    return len_best, d_min, label_best


def template_preprocessing(templates, alpha, normalize=True, normalization_type='z-score'):
    """
    Normalize the template sequences if required and save some information (length, minimum, and maximum) of each
    of the template sequences.

    :param templates: see function `segment_repeat_sequences`.
    :param alpha: see function `segment_repeat_sequences`.
    :param normalize: see function `segment_repeat_sequences`.
    :param normalization_type: see function `segment_repeat_sequences`.

    :return: (templates_norm, templates_info, min_length, max_length)
    - templates_norm: list of normalized template sequences with the same format as `templates`.
    - templates_info: list of namedtuples with information about the templates.
    - min_length: int value specifying the minimum length of the subsequence to be considered during matching.
    - max_length: int value specifying the maximum length of the subsequence to be considered during matching.
    """
    num_actions = len(templates)
    logger.info("Number of actions defined by the templates = %d.", num_actions)
    info_tuple = namedtuple('info_tuple', ['length', 'min', 'max', 'first_value', 'last_value'])
    if normalize:
        logger.info("Applying '%s' normalization to the template sequences.", normalization_type)

    min_length = np.inf
    max_length = -np.inf
    templates_norm = [[]] * num_actions
    templates_info = [[]] * num_actions
    for i in range(num_actions):
        num_templates = len(templates[i])
        logger.info("Number of templates for action %d = %d.", i + 1, num_templates)
        len_stats = np.percentile([a.shape[0] for a in templates[i]], [0, 50, 100])
        v = min(len_stats[0], max(2, np.floor(alpha * len_stats[1])))
        if v < min_length:
            min_length = v

        v = max(len_stats[2], np.ceil((1.0 / alpha) * len_stats[1]))
        if v > max_length:
            max_length = v

        templates_norm[i] = [[]] * num_templates
        templates_info[i] = [[]] * num_templates
        for j in range(num_templates):
            if normalize:
                if normalization_type == 'z-score':
                    arr = stats.zscore(templates[i][j])
                else:
                    arr = normalize_maxmin(templates[i][j])
            else:
                arr = templates[i][j]

            templates_norm[i][j] = arr
            templates_info[i][j] = info_tuple(
                length=arr.shape[0], min=np.min(arr), max=np.max(arr), first_value=arr[0, 0], last_value=arr[-1, 0]
            )

    return templates_norm, templates_info, int(min_length), int(max_length)


def segment_repeat_sequences(data, templates, normalize=True, normalization_type='z-score',
                             warping_window=None, alpha=0.75):
    """
    Segment the sequence `data` to closely match the sequences specified in the list `templates`.

    :param data: numpy array of shape (N, 1) with float values corresponding to the data sequence.
    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, 1)) corresponding to a template sequence.
    :param normalize: Apply normalization to the templates and the data subsequences, if set to True.
    :param normalization_type: Type of normalization to apply. Should be set to 'z-score' or 'max-min'.
    :param warping_window: Size of the warping window used to constrain the DTW matching path. This is also know as
                           the Sakoe-Chiba band in DTW literature. This can be set to `None` if no warping window
                           constraint is to be applied; else it should be set to a fractional value in (0, 1]. The
                           actual warping window is obtained by multiplying this fraction with the length of the
                           longer sequence. Suppose this window value is `w`, then any point `(i, j)` along the DTW
                           path satisfies `|i - j| <= w`. Setting this to a large value (closer to 1), allows the
                           warping path to be flexible, while setting it to a small value (closer to 0) will constrain
                           the warping path to be closer to the diagonal. Note that a small value can also speed-up
                           the DTW calculation significantly.
    :param alpha: float value in the range `(0, 1)`, but recommended to be in the range `[0.5, 0.8]`. This value
                  controls the search range for the subsequence length. If `m` is the median length of the template
                  sequences, then the search range for the subsequences is obtained by uniform sampling of the
                  interval `[alpha * m, (1 / alpha) * m]`. A smaller value of `alpha` increases the search interval
                  of the subsequence length resulting in a higher search time, but also a more extensive search
                  for the best match. On the other hand, a larger value of `alpha` (e.g. 0.8) will result in a
                  faster but less extensive search.

    :return:
        data_segments: list of segmented subsequences, each of which are numpy arrays of shape (m, 1).
        labels: list of best-matching template labels for the subsequences, where value `i` corresponds to the
                templates in position `i - 1` of the input list `templates`.
    """
    if normalization_type not in ('z-score', 'max-min'):
        raise ValueError("Invalid value '{}' for the parameter 'normalization_type'.".format(normalization_type))

    logger.info("Length of the input sequence = %d.", data.shape[0])
    templates_norm, templates_info, min_length, max_length = template_preprocessing(
        templates, alpha, normalize=normalize, normalization_type=normalization_type
    )

    # Starting from the left end of the sequence, find the subsequence with minimum average DTW distance from
    # the templates. Repeat this iteratively to find the segments
    logger.info("Search range for the subsequence length = [%d, %d].", min_length, max_length)
    data_segments = []
    labels = []
    data_rem = copy.copy(data)
    k = 1
    while data_rem.shape[0] > min_length:
        m, d_min, label = search_subsequence(data_rem, templates_norm, templates_info, min_length, max_length,
                                             normalize=normalize, normalization_type=normalization_type,
                                             warping_window=warping_window)
        data_segments.append(data_rem[:m, :])
        labels.append(label)
        data_rem = data_rem[m:, :]
        logger.info("Length of subsequence %d = %d. Matched template label = %d. Average DTW distance to the "
                    "matched templates = %.6f.", k, m, label, d_min)
        k += 1

    return data_segments, labels
