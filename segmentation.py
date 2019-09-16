"""
Segmenting a time series with multiple repetitions from a given set of actions.

Dynamic time warping (DTW) implementation by the DTAI research group, which has fast implementations (with C
bindings), is used. Documentation and installation instructions can be found below:
https://github.com/wannesm/dtaidistance
https://dtaidistance.readthedocs.io/en/latest/index.html

"""
import numpy as np
import copy
from collections import namedtuple
import multiprocessing
import pickle
from functools import partial
from scipy import stats
from itertools import combinations
from tslearn.metrics import njit_dtw
from tslearn.utils import to_time_series
import logging
from repeat_motion_segmentation.utils import (
    normalize_maxmin, num_templates_to_sample, get_warping_window
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template_info_tuple = namedtuple('template_info_tuple', ['length', 'min', 'max', 'first_value', 'last_value'])


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
            val += (np.sqrt(np.sum((sequence[0, :] - temp.first_value) ** 2) +
                            np.sum((sequence[-1, :] - temp.last_value) ** 2)) / (len_seq + temp.length))

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
    min_seq = np.min(sequence, axis=0)
    max_seq = np.max(sequence, axis=0)

    val_min = np.inf
    for i, t in enumerate(templates):
        val = 0.0
        for j, temp in enumerate(t):
            info = templates_info[i][j]
            # For the first and last values of the sequence, we calculate the exact deviation instead of
            # deviation with the maximum or minimum
            dev_first_last = np.sum((sequence[0, :] - info.first_value) ** 2) + \
                             np.sum((sequence[-1, :] - info.last_value) ** 2)

            # First lower bound calculated using the maximum and minimum values of the template as the envelope
            r1 = sequence - info.max
            r1[r1 < 0.0] = 0.0
            r1[0, :] = 0.0
            r1[-1, :] = 0.0

            r2 = info.min - sequence
            r2[r2 < 0.0] = 0.0
            r2[0, :] = 0.0
            r2[-1, :] = 0.0

            val1 = np.sqrt(np.sum(r1 ** 2) + np.sum(r2 ** 2) + dev_first_last) / (len_seq + info.length)

            # Second lower bound calculated using the maximum and minimum values of the sequence as the envelope
            r1 = temp - max_seq
            r1[r1 < 0.0] = 0.0
            r1[0, :] = 0.0
            r1[-1, :] = 0.0

            r2 = min_seq - temp
            r2[r2 < 0.0] = 0.0
            r2[0, :] = 0.0
            r2[-1, :] = 0.0

            val2 = np.sqrt(np.sum(r1 ** 2) + np.sum(r2 ** 2) + dev_first_last) / (len_seq + info.length)

            # Maximum of the two lower bounds is still a lower bound. This is added up to compute the average
            val += max(val1, val2)

        val /= len(t)
        if val < val_min:
            val_min = val

    return val_min


def average_distance_to_templates(sequence, templates, warping_window):
    sequence = to_time_series(sequence, remove_nans=True)
    len_seq = sequence.shape[0]
    val_min = np.inf
    label = 1
    for i, t in enumerate(templates, start=1):
        val = 0.0
        for temp in t:
            len_temp = temp.shape[0]
            if warping_window is None:
                mask = np.zeros((len_seq, len_temp))
            else:
                warping_window = get_warping_window(warping_window, len_seq, len_temp)
                mask = np.full((len_seq, len_temp), np.inf)
                ind_diff = np.abs(np.arange(len_seq)[:, np.newaxis] * np.ones((1, len_temp)) -
                                  np.ones((len_seq, 1)) * np.arange(len_temp)[np.newaxis, :])
                mask[ind_diff <= warping_window] = 0

            d = njit_dtw(sequence, to_time_series(temp, remove_nans=True), mask=mask) / float(len_seq + len_temp)
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

    if warping_window is None:
        mask = np.zeros((len_seq, len_temp))
    else:
        warping_window = get_warping_window(warping_window, len_seq, len_temp)
        mask = np.full((len_seq, len_temp), np.inf)
        ind_diff = np.abs(np.arange(len_seq)[:, np.newaxis] * np.ones((1, len_temp)) -
                          np.ones((len_seq, 1)) * np.arange(len_temp)[np.newaxis, :])
        mask[ind_diff <= warping_window] = 0

    d = (njit_dtw(to_time_series(sequence, remove_nans=True), to_time_series(t, remove_nans=True), mask=mask) /
         float(len_seq + len_temp))

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
            sequence_norm = (sequence[:m, :] - sequence_mean[m - 1, :]) / sequence_stdev[m - 1, :]
        else:
            # Max-min normalization
            mask = sequence_max[m - 1, :] > sequence_min[m - 1, :]
            sequence_norm = np.ones_like(sequence[:m, :])
            if np.any(mask):
                sequence_norm[:, mask] = ((sequence[:m, mask] - sequence_min[m - 1, mask]) /
                                          (sequence_max[m - 1, mask] - sequence_min[m - 1, mask]))

    else:
        sequence_norm = sequence[:m, :]

    return sequence_norm


def search_subsequence(sequence, templates, templates_info, min_length, max_length, normalize=True,
                       normalization_type='z-score', warping_window=None, use_lower_bounds=True):
    """
    Search for the subsequence that leads to minimum average DTW distance to the template sequences.

    :param sequence: numpy array of shape (N, d) with float values.
    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, d)) corresponding to a template sequence.
    :param templates_info: list similar to `templates`, but each element of the list is a namedtuple with information
                           about the template such as length, minimum value, and maximum value.
    :param min_length: minimum length of the subsequence to search.
    :param max_length: maximum length of the subsequence to search.
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
            den = np.arange(1, N + 1).astype(np.float)[:, np.newaxis]
            sequence_mean = np.cumsum(sequence, axis=0) / den
            arr = (sequence - sequence_mean) ** 2
            sequence_stdev = np.sqrt(np.clip(np.cumsum(arr, axis=0) / den, 1e-16, None))
        else:
            # Calculate the rolling maximum and minimum of the entire data sequence. This will be used in
            # the max-min normalization of subsequences of different lengths
            sequence_min = np.minimum.accumulate(sequence, axis=0)
            sequence_max = np.maximum.accumulate(sequence, axis=0)

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


def helper_average_dtw_distance(sequences, warping_window, indices):
    d_avg, _ = average_distance_to_templates(sequences[indices[0]], [[sequences[i] for i in indices[1:]]],
                                             warping_window)
    return d_avg


def find_distance_thresholds(templates, templates_info, warping_window, max_num_samples=10000,
                             upper_quantile=1.0, seed=1234):
    """
    For each action category, we find an upper threshold on the average DTW distance that will help filter out
    segments of the time series that are bad matches.

    To do this, we capture the empirical distribution of the average DTW distance between a template sequence and a
    set of template sequences (from the same action). If there are `n` templates, first we find a smaller number `k`
    of templates so that we can compute a sufficient number of average distance samples. If we consider one out of
    the `n` templates, then we can select `k` templates from the remaining `n - 1` templates in `(n - 1)_C_k` ways.
    The average DTW distance between the single template and the set of `k` templates can be calculated in each case.
    This can be repeated `n` times by holding out a different template each time, giving a total of  `n (n - 1)_C_k`
    average DTW distance values. If `n` is sufficiently large, we can get enough samples to capture the distribution
    of the average DTW distance. Listed below is a sequence of `n` and `k` values and the number of distance samples
    it would produce:
    n = 4, best k = 2, #samples = 12
    n = 5, best k = 2, #samples = 30
    n = 6, best k = 3, #samples = 60
    n = 7, best k = 3, #samples = 140
    n = 8, best k = 4, #samples = 280
    n = 9, best k = 4, #samples = 630
    n = 10, best k = 5, #samples = 1260
    n = 11, best k = 5, #samples = 2772
    n = 12, best k = 6, #samples = 5544
    n = 13, best k = 6, #samples = 12012
    n = 14, best k = 7, #samples = 24024
    n = 15, best k = 7, #samples = 51480

    The 99-th percentile of the distances is calculated as the upper threshold.

    :param templates: see function `segment_repeat_sequences`.
    :param templates_info: see function `segment_repeat_sequences`.
    :param warping_window: see function `segment_repeat_sequences`.
    :param max_num_samples: If `n` is larger than 13, the number of combinations can become very large. This sets an
                            upper bound on the number of distance samples to be computed.
    :param upper_quantile: Upper quantile used for the threshold calculation.
    :param seed: Seed of the random number generator.

    :return: (distance_thresholds, templates_selected, templates_info_selected)
    - distance_thresholds: List of distance thresholds, one for each action.
    - templates_selected: Same format as `templates`, but includes only `k` randomly selected templates per action.
    - templates_info_selected: Same format as `templates_info`, but includes only `k` randomly selected templates
                               per action.
    """
    np.random.seed(seed)
    num_proc = max(1, multiprocessing.cpu_count() - 1)

    templates_selected = []
    templates_info_selected = []
    distance_thresholds = []
    num_templates = dict()
    for i in range(len(templates)):
        # number of templates for this action
        n = len(templates[i])
        if n in num_templates:
            k, ns = num_templates[n]
        else:
            k, ns = num_templates_to_sample(n)
            num_templates[n] = (k, ns)

        logger.info("Action %d:", i + 1)
        logger.info("Selecting %d out of %d templates for matching based on average DTW distance", k, n)
        ind = np.random.permutation(n)[:k]
        templates_selected.append([templates[i][j] for j in ind])
        templates_info_selected.append([templates_info[i][j] for j in ind])

        comb_list = list(combinations(range(n - 1), k))
        len_comb_list = len(comb_list)
        a = int(np.ceil(float(max_num_samples) / n))
        if len_comb_list > a:
            comb_list = [comb_list[j] for j in np.random.permutation(len_comb_list)[:a]]

        if num_proc > 1:
            index_list = []
            for j in range(n):
                # Every index excluding `j`
                ind = [jj for jj in range(n) if jj != j]
                index_list.extend([[j] + [ind[t] for t in tup] for tup in comb_list])

            helper_partial = partial(helper_average_dtw_distance, templates[i], warping_window)
            pool_obj = multiprocessing.Pool(processes=num_proc)
            distances = []
            _ = pool_obj.map_async(helper_partial, index_list, chunksize=None, callback=distances.extend)
            pool_obj.close()
            pool_obj.join()
        else:
            distances = []
            for j in range(n):
                # Every index excluding `j`
                ind = [jj for jj in range(n) if jj != j]
                distances.extend([
                    helper_average_dtw_distance(templates[i], warping_window, [j] + [ind[t] for t in tup])
                    for tup in comb_list
                ])

        distances = np.array(distances)
        if distances.shape[0] < 100:
            logger.warning("Sample size of distances (%d) may be too small for reliable threshold estimation.",
                           distances.shape[0])

        distance_thresholds.append(np.percentile(distances, 100 * upper_quantile))
        logger.info("Upper threshold on distance DTW distance = %.6f", distance_thresholds[-1])
        v = np.percentile(distances, [0, 50, 100])
        logger.info("Min = %.6f, Median = %.6f, Max = %.6f", v[0], v[1], v[2])

    return distance_thresholds, templates_selected, templates_info_selected


def normalize_templates(templates, alpha, normalize=True, normalization_type='z-score'):
    """
    Normalize the template sequences if required and save some information (length, minimum, and maximum) of each
    of the template sequences.

    :param templates: see function `segment_repeat_sequences`.
    :param alpha: see function `segment_repeat_sequences`.
    :param normalize: see function `segment_repeat_sequences`.
    :param normalization_type: see function `segment_repeat_sequences`.

    :return: (templates_norm, templates_info, length_stats)
    - templates_norm: list of normalized template sequences with the same format as `templates`.
    - templates_info: list of namedtuples with information about the templates.
    - length_stats: tuple `(min_length, median_length, max_length)`, where
        -- min_length: minimum length of the subsequence to be considered during matching.
        -- max_length: maximum length of the subsequence to be considered during matching.
        -- median_length: median length of the template sequences across all actions.
    """
    num_actions = len(templates)
    logger.info("Number of actions defined by the templates = %d.", num_actions)
    if normalize:
        logger.info("Applying '%s' normalization to the template sequences.", normalization_type)

    min_length = np.inf
    max_length = -np.inf
    templates_norm = [[]] * num_actions
    templates_info = [[]] * num_actions
    templates_length = []
    for i in range(num_actions):
        num_templates = len(templates[i])
        # logger.info("Number of templates for action %d = %d.", i + 1, num_templates)
        len_arr = [a.shape[0] for a in templates[i]]
        templates_length.extend(len_arr)
        len_stats = np.percentile(len_arr, [0, 50, 100])

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
                    arr = stats.zscore(templates[i][j], axis=0)
                else:
                    arr = normalize_maxmin(templates[i][j])
            else:
                arr = templates[i][j]

            templates_norm[i][j] = arr
            templates_info[i][j] = template_info_tuple(
                length=arr.shape[0], min=np.min(arr, axis=0), max=np.max(arr, axis=0),
                first_value=arr[0, :], last_value=arr[-1, :]
            )

    length_stats = [int(min_length), int(np.percentile(templates_length, 50)), int(max_length)]

    return templates_norm, templates_info, length_stats


def preprocess_templates(templates, normalize=True, normalization_type='z-score', warping_window=None, alpha=0.75,
                         templates_results_file=None):
    """
    Normalize the template sequences and calculate thresholds on the average DTW distance.

    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, d)) corresponding to a template sequence.
    :param normalize: see function `segment_repeat_sequences`.
    :param normalization_type: see function `segment_repeat_sequences`.
    :param warping_window: see function `segment_repeat_sequences`.
    :param alpha: float value in the range `(0, 1)`, but recommended to be in the range `[0.5, 0.8]`. This value
                  controls the search range for the subsequence length. If `m` is the median length of the template
                  sequences, then the search range for the subsequences is obtained by uniform sampling of the
                  interval `[alpha * m, (1 / alpha) * m]`. A smaller value of `alpha` increases the search interval
                  of the subsequence length resulting in a higher search time, but also a more extensive search
                  for the best match. On the other hand, a larger value of `alpha` (e.g. 0.8) will result in a
                  faster but less extensive search.
    :param templates_results_file: Filename for the pickle file in which the processed template results will be
                                   saved. This can be used to avoid processing the templates (which can be time
                                   consuming) repeatedly on multiple runs.

    :return results: A dict with the following items:
    {'templates_normalized', 'templates_info', 'distance_thresholds', 'length_stats'}
    - templates_normalized: list of selected normalized template sequences with the same format as the input
                            `templates`. Only `k` out of `n` templates are selected per action.
    - templates_info: list of namedtuples with information about the selected templates.
    - distance_thresholds: list of upper thresholds on the average DTW distance, one for each action.
    - length_stats: see function `normalize_templates`.
    """
    if normalization_type not in ('z-score', 'max-min'):
        raise ValueError("Invalid value '{}' for the parameter 'normalization_type'.".format(normalization_type))

    templates_norm, templates_info, length_stats = normalize_templates(
        templates, alpha, normalize=normalize, normalization_type=normalization_type
    )

    logger.info("Calculating the upper threshold on the average DTW distance for each action based on the "
                "given template sequences.")
    distance_thresholds, templates_norm_selected, templates_info_selected = find_distance_thresholds(
        templates_norm, templates_info, warping_window
    )

    results = {
        'templates_normalized': templates_norm_selected,
        'templates_info': templates_info_selected,
        'distance_thresholds': distance_thresholds,
        'length_stats': length_stats
    }
    if templates_results_file:
        logger.info("Preprocessed template results have been saved to the file: %s", templates_results_file)
        with open(templates_results_file, 'wb') as fp:
            pickle.dump(results, fp)

    return results


def segment_repeat_sequences(data, templates_norm, templates_info, distance_thresholds, length_stats,
                             normalize=True, normalization_type='z-score', warping_window=None):
    """
    Segment the sequence `data` to closely match the sequences specified in the list `templates_norm`.

    :param data: numpy array of shape (N, d) with float values corresponding to the data sequence.
                 `N` is the number of points in the series and `d` is the dimension of each point in the series.
    :param templates_norm: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`,
                           and each `s_ij` is a numpy array (of shape (M, d)) corresponding to a template sequence.
                           The template sequences are expected to be normalized.
    :param templates_info: list of namedtuples with information about the templates in `templates_norm`.
    :param distance_thresholds: list of float values with length equal to the number of the actions (i.e. length
                                of `templates_norm`). Each value is an upper threshold on the average DTW distance
                                corresponding to templates from a given action.
    :param length_stats: tuple `(min_length, median_length, max_length)` referring to the length of the
                         template subsequences.
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

    :return: (data_segments, labels)
        - data_segments: list of segmented subsequences, each of which are numpy arrays of shape (m, d) (`m` can be
                         different for each subsequence).
        - labels: list of best-matching template labels for the subsequences, where value `i` corresponds to the
                  templates in position `i - 1` of the input list `templates`. Label value `0` indicates that the
                  corresponding subsequence in `data_segments` could not be matched to any action.
    """
    logger.info("Length of the input sequence = %d. Dimension of the input sequence = %d.",
                data.shape[0], data.shape[1])
    min_length, median_length, max_length = length_stats
    logger.info("Search range for the subsequence length = [%d, %d].", min_length, max_length)

    # Starting from the left end of the sequence, find the subsequences with minimum average DTW distance from
    # the templates. Repeat this iteratively to extract the segments
    data_segments = []
    labels = []
    num_seg = 0
    data_rem = copy.copy(data)
    while data_rem.shape[0] > min_length:
        offset = 0
        info_best = [0, 0, np.inf, 0]       # offset, sequence_length, avg_distance, label
        match = False
        while (data_rem.shape[0] - offset) > min_length:
            m, d_avg, label = search_subsequence(
                data_rem[offset:, :], templates_norm, templates_info, min_length, max_length, normalize=normalize,
                normalization_type=normalization_type, warping_window=warping_window
            )
            if d_avg <= distance_thresholds[label - 1]:
                if d_avg < info_best[2]:
                    # Best matching subsequence so far
                    match = True
                    info_best = [offset, m, d_avg, label]

            offset += 1
            logger.info("offset = %d, match = %d", offset, int(match))
            if match and (offset - info_best[0]) > 5:
                # If a match has been found for a certain offset value, and a string of increasing values of
                # the offset does not lead to a better match (lower average DTW), then we break in order to
                # speed up the search. The choice of 5 is heuristic
                break

        if match:
            num_seg += 1
            offset, m, d_avg, label = info_best
            if offset > 0:
                # The segment prior to the offset does not match any action. Hence, its label is set to 0
                data_segments.append(data_rem[:offset, :])
                labels.append(0)

            data_segments.append(data_rem[offset:(offset + m), :])
            labels.append(label)
            data_rem = data_rem[(offset + m):, :]
            logger.info("Segment %d: Length of matched segment = %d, offset = %d, matched template label = %d, "
                        "average DTW distance = %.6f.", num_seg, m, offset, label, d_avg)
        else:
            # No matches could be found in this subsequence
            data_segments.append(data_rem)
            labels.append(0)
            data_rem = np.zeros((0, data.shape[1]))
            break

    if data_rem.shape[0] > 0:
        # Any remaining unmatched segment
        data_segments.append(data_rem)
        labels.append(0)

    return data_segments, labels
