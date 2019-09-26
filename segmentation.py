"""
Segmenting a time series with multiple repetitions from a given set of actions.

Dynamic time warping (DTW) implementation from the `tslearn` library is used. It supports multivariate time series
and has fast a implementation based on numba.

https://tslearn.readthedocs.io/en/latest/index.html
https://github.com/rtavenar/tslearn

"""
import numpy as np
import copy
from collections import namedtuple
import multiprocessing
import pickle
import operator
from functools import partial
from scipy import stats
from itertools import combinations
from numba import jit, prange
from tslearn.metrics import njit_dtw
import logging
from repeat_motion_segmentation.utils import (
    find_max_combinations,
    sakoe_chiba_mask,
    stratified_sample
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template_info_tuple = namedtuple('template_info_tuple', ['length', 'min', 'max', 'first_value', 'last_value'])


@jit(nopython=True)
def lb_distance_to_templates1(sequence, templates_info):
    """
    Simple but fast lower bound to the DTW based on comparison of only the first and last values of the sequence
    and the templates.
    """
    len_seq = sequence.shape[0]
    val_min = np.inf
    for t in templates_info:
        for temp in t:
            val = (np.sqrt(np.sum((sequence[0, :] - temp.first_value) ** 2) +
                           np.sum((sequence[-1, :] - temp.last_value) ** 2)) / (len_seq + temp.length))

            if val < val_min:
                val_min = val

    return val_min


@jit(nopython=True, parallel=False)
def lb_distance_to_templates2(sequence, templates, templates_info):
    """
    Minimum distance to the templates, where the distance is the lower bound of the DTW distance proposed by:
    Yi, Byoung-Kee, H. V. Jagadish, and Christos Faloutsos. "Efficient retrieval of similar time sequences
    under time warping." Proceedings 14th International Conference on Data Engineering. IEEE, 1998.

    Minor modifications are made to the method proposed by Yi et. al. The exact squared difference between the first
    and the last values are computed, since that is a requirement for DTW distance. The minimum-maximum envelope
    of the template sequence is used to calculate the first lower bound, and the second lower bound is calculated
    by using the minimum-maximum envelope of the input sequence. The maximum of both these values is taken as the
    final lower bound.
    """
    len_seq = sequence.shape[0]
    dim = sequence.shape[1]

    # Maximum and minimum of the sequence along each dimension.
    # Not using `axis=0` argument for the numpy.min and numpy.max functions because it is not supported by numba
    min_seq = np.zeros(dim)
    max_seq = np.zeros(dim)
    for j in prange(dim):
        min_seq[j] = np.min(sequence[:, j])
        max_seq[j] = np.max(sequence[:, j])

    val_min = np.inf
    for i, t in enumerate(templates):
        for j, temp in enumerate(t):
            info = templates_info[i][j]
            # For the first and last values of the sequence, we calculate the exact deviation instead of
            # deviation with the maximum or minimum
            dev_first_last = np.sum((sequence[0, :] - info.first_value) ** 2) + \
                             np.sum((sequence[-1, :] - info.last_value) ** 2)
            diff1 = 0.
            diff2 = 0.
            for k in prange(dim):
                # First difference uses the maximum and minimum values of the template sequence as the envelope
                vec1 = sequence[:, k] - info.max[k]
                vec1[vec1 < 0.] = 0.
                vec2 = info.min[k] - sequence[:, k]
                vec2[vec2 < 0.] = 0.
                diff1 += np.sum(vec1[1:-1] ** 2 + vec2[1:-1] ** 2)

                # Second difference uses the maximum and minimum values of the input sequence as the envelope
                vec3 = temp[:, k] - max_seq[k]
                vec3[vec3 < 0.] = 0.
                vec4 = min_seq[k] - temp[:, k]
                vec4[vec4 < 0.] = 0.
                diff2 += np.sum(vec3[1:-1] ** 2 + vec4[1:-1] ** 2)

            val = np.sqrt(max(diff1, diff2) + dev_first_last) / (len_seq + info.length)
            if val < val_min:
                val_min = val

    return val_min


@jit(nopython=True, parallel=False)
def fast_approx_matching(sequence, templates, templates_info, template_counts, feature_mask_per_action,
                         warping_window, max_length_deviation, dist_min_prior=np.inf):
    len_seq = sequence.shape[0]
    dim = sequence.shape[1]

    # Maximum and minimum of the sequence along each dimension.
    # Not using `axis=0` argument for the numpy.min and numpy.max functions because it is not supported by numba
    min_seq = np.zeros(dim)
    max_seq = np.zeros(dim)
    for j in prange(dim):
        min_seq[j] = np.min(sequence[:, j])
        max_seq[j] = np.max(sequence[:, j])

    dist_min = dist_min_prior
    label = 0
    for i in range(template_counts[0]):     # each action
        mask_feat = feature_mask_per_action[i]
        ind_feat = np.arange(dim)[mask_feat]
        for j in range(template_counts[i + 1]):     # each template corresponding to this action
            temp = templates[i][j]
            temp_info = templates_info[i][j]
            # Heuristic to speedup
            if abs(len_seq - temp_info.length) > max_length_deviation[i]:
                continue

            v = (sequence[0, :] - temp_info.first_value) ** 2 + (sequence[-1, :] - temp_info.last_value) ** 2
            dev_first_last = np.sum(v[mask_feat])

            # First simple (but loose) lower bound to the DTW distance
            val_lb1 = np.sqrt(dev_first_last) / (len_seq + temp_info.length)
            if val_lb1 > dist_min:
                # Lower bound is greater than the current minimum. So the actual DTW distance need not be calculated
                continue

            # Second lower bound to the DTW distance
            diff1 = 0.
            diff2 = 0.
            for k in ind_feat:
                # First difference uses the maximum and minimum values of the template sequence as the envelope
                vec1 = sequence[:, k] - temp_info.max[k]
                vec1[vec1 < 0.] = 0.
                vec2 = temp_info.min[k] - sequence[:, k]
                vec2[vec2 < 0.] = 0.
                diff1 += np.sum(vec1[1:-1] ** 2 + vec2[1:-1] ** 2)

                # Second difference uses the maximum and minimum values of the input sequence as the envelope
                vec3 = temp[:, k] - max_seq[k]
                vec3[vec3 < 0.] = 0.
                vec4 = min_seq[k] - temp[:, k]
                vec4[vec4 < 0.] = 0.
                diff2 += np.sum(vec3[1:-1] ** 2 + vec4[1:-1] ** 2)

            val_lb2 = np.sqrt(max(diff1, diff2) + dev_first_last) / (len_seq + temp_info.length)
            if val_lb2 > dist_min:
                # Lower bound is greater than the current minimum. So the actual DTW distance need not be calculated
                continue

            # DTW distance calculation
            if warping_window is None:
                mask = np.zeros((len_seq, temp_info.length))
            else:
                mask = sakoe_chiba_mask(len_seq, temp_info.length, warping_window)

            val = njit_dtw(sequence[:, mask_feat], temp[:, mask_feat], mask=mask) / float(len_seq + temp_info.length)
            if val < dist_min:
                dist_min = val
                label = i + 1

    return dist_min, label


# @jit(nopython=True)
def search_subsequence(sequence, templates, templates_info, template_counts, feature_mask_per_action, min_length,
                       max_length, max_length_deviation, normalize=True, warping_window=None, length_step=1,
                       dist_min_prior=np.inf):
    """
    Search for the subsequence that leads to minimum average DTW distance to the template sequences.

    :param sequence: numpy array of shape (N, d) with float values.
    :param templates: tuple `[L_1, . . ., L_k]`, where each `L_i` is another tuple `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, d)) corresponding to a template sequence.
    :param templates_info: tuple similar to `templates`, but each element of the tuple is a namedtuple with
                           information about the template such as length, minimum value, and maximum value.
    :param template_counts: list or numpy array of integer values. First value is the number of actions. The rest of
                            the values are the number of templates correspondsing to each action.
    :param feature_mask_per_action: see `segment_repeat_sequences`.
    :param min_length: minimum length of the subsequence to search.
    :param max_length: maximum length of the subsequence to search.
    :param max_length_deviation: list with the maximum length deviation between template sequences from each action.
    :param normalize: see `segment_repeat_sequences`.
    :param warping_window: see `segment_repeat_sequences`.
    :param length_step: see `segment_repeat_sequences`.
    :param dist_min_prior: (float) Minimum distance found from a previous search or iteration.
                           Set to np.inf by default.

    :return: tuple `(len_best, d_min, label_best)`, where
             -- `len_best` is the best subsequence length,
             -- `d_min` is the minimum average DTW distance between the subsequence and the templates from
                 the matched action,
             -- `label_best` is the label of the best-matching template.
    """
    N = sequence.shape[0]
    if N > max_length:
        # Truncate the sequence at `max_length` since the rest of the sequence is not needed
        sequence = sequence[:max_length]
        N = max_length
    else:
        max_length = N

    if min_length >= max_length:
        min_length = max(1, max_length - 1)

    if normalize:
        # Calculate the rolling mean and standard-deviation of the entire data sequence. This will be used
        # for z-score normalization of subsequences of different lengths.
        """
        # Use this approach if numba (the @jit decorator) is used
        dim = sequence.shape[1]
        den = np.arange(1, N + 1)
        sequence_mean = np.zeros((N, dim))
        sequence_stdev = np.zeros((N, dim))
        for j in prange(dim):
            sequence_mean[:, j] = np.cumsum(sequence[:, j]) / den
            sequence_stdev[:, j] = np.sqrt(1e-16 +
                                           np.cumsum((sequence[:, j] - sequence_mean[:, j]) ** 2) / den)
        """
        # Using this approach causes `numba` to fail because `np.cumsum` is used with the argument `axis=0`.
        # Use it if not using the @jit decorator
        den = np.arange(1, N + 1).reshape((N, 1))
        sequence_mean = np.cumsum(sequence, axis=0) / den
        arr = (sequence - sequence_mean) ** 2
        sequence_stdev = np.sqrt(1e-16 + np.cumsum(arr, axis=0) / den)
    else:
        # This will not be used
        sequence_mean = sequence
        sequence_stdev = sequence

    length_range = np.random.permutation(np.arange(min_length, max_length + 1, length_step))
    len_best = length_range[0]
    label_best = 0
    dist_min = dist_min_prior
    for m in length_range:
        if normalize:
            sequence_norm = (sequence[:m, :] - sequence_mean[m - 1, :]) / sequence_stdev[m - 1, :]
        else:
            sequence_norm = sequence[:m, :]

        d, label = fast_approx_matching(sequence_norm, templates, templates_info, template_counts,
                                        feature_mask_per_action, warping_window, max_length_deviation,
                                        dist_min_prior=dist_min)
        if d < dist_min and label > 0:
            dist_min = d
            label_best = label
            len_best = m

    return len_best, dist_min, label_best


@jit(nopython=True)
def distance_to_templates(sequence, templates, warping_window, mask_features=None, add_noise=False):
    len_seq = sequence.shape[0]
    if add_noise:
        for j in prange(sequence.shape[1]):
            v = sequence[:, j]
            sequence[:, j] = v + 1e-5 * np.ptp(v) * np.random.randn(len_seq)

    val_min = np.inf
    label = 0
    for i, t in enumerate(templates):
        for temp in t:
            len_temp = temp.shape[0]
            if warping_window is None:
                mask = np.zeros((len_seq, len_temp))
            else:
                mask = sakoe_chiba_mask(len_seq, len_temp, warping_window)

            if mask_features is None:
                val = njit_dtw(sequence, temp, mask=mask) / float(len_seq + len_temp)
            else:
                val = (njit_dtw(sequence[:, mask_features], temp[:, mask_features], mask=mask) /
                       float(len_seq + len_temp))

            if val < val_min:
                val_min = val
                label = i + 1

    return val_min, label


def score_feature_selection(templates_same, templates_diff, warping_window, dim, index_features):
    mask_features = np.zeros(dim, dtype=np.bool)
    mask_features[list(index_features)] = True
    cnt_same = len(templates_same)
    score = 0.
    for i in range(cnt_same):
        templates_same_exc = (tuple([templates_same[j] for j in range(cnt_same) if j != i]), )
        dist_same, _ = distance_to_templates(templates_same[i], templates_same_exc, warping_window,
                                             mask_features=mask_features)
        dist_diff, _ = distance_to_templates(templates_same[i], templates_diff, warping_window,
                                             mask_features=mask_features)
        score += (dist_diff / max(dist_same, 1e-16))

    return index_features, score / cnt_same


def find_best_feature_subset(templates, warping_window=None, add_noise_sequences=True, bypass=False):
    """
    Find the best subset of features to include for each action. Since this is doing a search over all possible
    feature subsets, it should be used only for small number of features (less than 10).

    :param templates: see function `segment_repeat_sequences`.
    :param warping_window: see function `segment_repeat_sequences`.
    :param add_noise_sequences: Set to True in order to include some noise sequences to the feature selection.
    :param bypass: a boolean flag to bypass the feature selection step and include all features.

    :return feature_mask_per_action: tuple of boolean numpy arrays of length equal to the number of actions.
                                     Each boolean array has length equal to the number of features and indicates
                                     whether a feature is included or not for that action.
    """
    dim = templates[0][0].shape[1]
    num_actions = len(templates)
    if dim == 1:
        return tuple([np.array([True]) for _ in range(num_actions)])

    if bypass:
        mask = np.ones(dim, dtype=np.bool)
        return tuple([mask for _ in range(num_actions)])

    num_proc = max(1, multiprocessing.cpu_count() - 1)
    """
    # List all possible feature subsets of size >= 1
    feature_subsets = []
    ind = range(dim)
    for sz in range(1, dim + 1):
        feature_subsets.extend(combinations(ind, sz))
    """
    feature_subsets = [(j, ) for j in range(dim)]

    noise_sequences = []
    if add_noise_sequences:
        num_noise_sequences = len(templates[0])

        # Length of all the template sequences
        length_arr = np.array([b.shape[0] for a in templates for b in a], dtype=np.int)
        # The length of the noise sequences will be randomly sampled (with replacement) from `length_arr`
        length_noise_sequences = np.random.choice(length_arr, size=num_noise_sequences, replace=True)

        for n in range(num_noise_sequences):
            # Random Gaussian noise with standard deviation 0.001
            x = 0.001 * np.random.randn(length_noise_sequences[n], dim)
            noise_sequences.append(x)

    feature_mask_per_action = [None] * num_actions
    for i in range(num_actions):
        # Templates from the same action
        templates_same = tuple(templates[i])

        # Templates from different actions including noise sequences (if required).
        # Using tuples because list of lists causes errors with numba
        if add_noise_sequences:
            templates_diff = tuple([tuple(templates[j]) for j in range(num_actions) if j != i] +
                                   [tuple(noise_sequences)])
        else:
            templates_diff = tuple([tuple(templates[j]) for j in range(num_actions) if j != i])

        if num_proc > 1:
            helper_partial = partial(score_feature_selection, templates_same, templates_diff, warping_window, dim)
            pool_obj = multiprocessing.Pool(processes=num_proc)
            scores = []
            _ = pool_obj.map_async(helper_partial, feature_subsets, chunksize=None, callback=scores.extend)
            pool_obj.close()
            pool_obj.join()
        else:
            scores = [score_feature_selection(templates_same, templates_diff, warping_window, dim, ind)
                      for ind in feature_subsets]

        # Sort the features in decreasing order of score and find the normalized cumulative score.
        # The normalized cumulative score is thresholded to select the top ranked features
        scores_sorted = sorted(scores, key=operator.itemgetter(1), reverse=True)
        u = np.cumsum([t[1] for t in scores_sorted])
        v = (1. / u[-1]) * u
        logger.info("Action %d:", i + 1)
        logger.info("Ranked features and their normalized cumulative scores:")
        for j in range(dim):
            logger.info("%d, %.6f", scores_sorted[j][0][0], v[j])

        ind_selected = [scores_sorted[j][0][0] for j in range(dim // 2)]
        logger.info("Selected features = (%s).", ', '.join(map(str, ind_selected)))
        logger.info("")
        feature_mask_per_action[i] = np.zeros(dim, dtype=np.bool)
        feature_mask_per_action[i][ind_selected] = True

    return tuple(feature_mask_per_action)


def helper_distance_to_templates(sequences, mask_features, warping_window, indices):
    d_min, _ = distance_to_templates(
        sequences[indices[0]], (tuple([sequences[i] for i in indices[1:]]), ), warping_window,
        mask_features=mask_features
    )
    return d_min


def find_distance_thresholds(templates, template_labels, templates_info, feature_mask_per_action, warping_window,
                             num_templates_to_select=None, max_num_samples=10000):
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

    :param templates: see function `preprocess_templates`.
    :param template_labels: see function `preprocess_templates`.
    :param templates_info: see function `segment_repeat_sequences`.
    :param feature_mask_per_action: tuple of length equal to the number of actions, where each item is a boolean
                                    numpy array that acts a feature mask (selected features for a given action).
    :param warping_window: see function `segment_repeat_sequences`.
    :param num_templates_to_select: Number of templates to select. If set to None, this is automatically set to
                                    a suitable value.
    :param max_num_samples: If `n` is larger than 13, the number of combinations can become very large. This sets an
                            upper bound on the number of distance samples to be computed.

    :return: (distance_thresholds, templates_selected, template_labels_selected, templates_info_selected,
              template_counts)
    - distance_thresholds: List of distance thresholds, one for each action.
    - templates_selected: Selected subset of normalized template sequences per action. A tuple of tuples, where each
                          element of inner tuple is a 2d numpy array with the template sequences.
    - template_labels_selected: Labels of the selected template sequences per action. A tuple of tuples, where each
                                element of the inner tuple is the label.
    - templates_info_selected: Information about the selected subset of template sequences per action. A tuple of
                               tuples, where each element of inner tuple is a namedtuple of type `template_info_tuple`.
    - template_counts: array of integers, where the first value is the number of actions and the subsequent values
                       are the number of templates per action.
    """
    num_proc = max(1, multiprocessing.cpu_count() - 1)
    num_actions = len(templates)

    templates_selected = []
    template_labels_selected = []
    templates_info_selected = []
    template_counts = [num_actions]
    distance_thresholds = []
    for i in range(num_actions):
        # number of templates for this action
        n = len(templates[i])
        k, _ = find_max_combinations(n, k=num_templates_to_select)
        logger.info("Action %d:", i + 1)
        logger.info("Selecting %d out of %d templates for matching based on DTW distance", k, n)

        labels_group = np.array([tup[1] for tup in template_labels[i]])
        ind = stratified_sample(labels_group, n, k)
        templates_selected.append(tuple([templates[i][j] for j in ind]))
        template_labels_selected.append(tuple([template_labels[i][j] for j in ind]))
        templates_info_selected.append(tuple([templates_info[i][j] for j in ind]))
        template_counts.append(ind.shape[0])

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

            helper_partial = partial(helper_distance_to_templates, templates[i], feature_mask_per_action[i],
                                     warping_window)
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
                    helper_distance_to_templates(templates[i], feature_mask_per_action[i], warping_window,
                                                 [j] + [ind[t] for t in tup]) for tup in comb_list
                ])

        distances = np.array(distances)
        if distances.shape[0] < 100:
            logger.warning("Sample size of distances (%d) may be too small for reliable threshold estimation.",
                           distances.shape[0])

        # A value slightly larger than the maximum distance and a the 1.5 IQR rule are calculated. The larger of the
        # two is used as the threshold
        v = np.percentile(distances, [0, 25, 50, 75, 100])
        th = max(1.01 * v[4], v[3] + 1.5 * (v[3] - v[1]))
        distance_thresholds.append(th)
        logger.info("Upper threshold on the DTW distance to templates = %.6f", distance_thresholds[-1])
        logger.info("Min = %.6f, Median = %.6f, Max = %.6f", v[0], v[2], v[4])

    # Converting to tuple since it helps with numba compilation in `nopython` mode
    templates_selected = tuple(templates_selected)
    template_labels_selected = tuple(template_labels_selected)
    templates_info_selected = tuple(templates_info_selected)
    template_counts = np.array(template_counts, dtype=np.int)

    return distance_thresholds, templates_selected, template_labels_selected, templates_info_selected, template_counts


def normalize_templates(templates, normalize=True):
    """
    Normalize the template sequences if required and save some information (length, minimum, and maximum) of each
    of the template sequences.

    :param templates: see function `segment_repeat_sequences`.
    :param normalize: see function `segment_repeat_sequences`.

    :return: (templates_norm, templates_info, length_stats)
    - templates_norm: list of normalized template sequences with the same format as `templates`.
    - templates_info: list of namedtuples with information about the templates.
    - length_stats: tuple `(min_length, median_length, max_length, max_length_deviation)`, where
        -- min_length: minimum length of the subsequence to be considered during matching.
        -- max_length: maximum length of the subsequence to be considered during matching.
        -- median_length: median length of the template sequences across all actions.
        -- max_length_deviation: list with the maximum length deviation between template sequences from each action.
    """
    num_actions = len(templates)
    logger.info("Number of actions defined by the templates = %d.", num_actions)

    min_length = np.inf
    max_length = -np.inf
    max_length_deviation = [0] * num_actions
    templates_norm = [[]] * num_actions
    templates_info = [[]] * num_actions
    templates_length = []
    for i in range(num_actions):
        num_templates = len(templates[i])
        len_arr = [a.shape[0] for a in templates[i]]
        templates_length.extend(len_arr)
        len_stats = np.percentile(len_arr, [0, 25, 50, 75, 100])

        max_length_deviation[i] = int(np.ceil(1.1 * (len_stats[-1] - len_stats[0])))

        # 1.5 times the interquartile range is used as the deviation
        delta = 1.5 * (len_stats[3] - len_stats[1])
        v = min(len_stats[0], max(2, len_stats[1] - delta))
        if v < min_length:
            min_length = v

        v = max(len_stats[-1], len_stats[3] + delta)
        if v > max_length:
            max_length = v

        templates_norm[i] = [[]] * num_templates
        templates_info[i] = [[]] * num_templates
        for j in range(num_templates):
            if normalize:
                arr = stats.zscore(templates[i][j], axis=0)
            else:
                arr = templates[i][j]

            templates_norm[i][j] = arr
            templates_info[i][j] = template_info_tuple(
                length=arr.shape[0], min=np.min(arr, axis=0), max=np.max(arr, axis=0),
                first_value=arr[0, :], last_value=arr[-1, :]
            )

    logger.info("Maximum length deviation between templates from each action: %s",
                ', '.join(map(str, max_length_deviation)))
    length_stats = [int(min_length), int(np.percentile(templates_length, 50)), int(max_length),
                    max_length_deviation]

    return templates_norm, templates_info, length_stats


def preprocess_templates(templates, template_labels, normalize=True, warping_window=None,
                         num_templates_to_select=None, templates_results_file=None, seed=1234):
    """
    Normalize the template sequences and calculate thresholds on the average DTW distance.

    :param templates: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and
                      each `s_ij` is a numpy array (of shape (M, d)) corresponding to a template sequence.
    :param template_labels: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`,
                            and each `s_ij` is a tuple corresponding to a template sequence. The tuple consists of
                            the label and an additional category, e.g. the speed of rotation.
    :param normalize: see function `segment_repeat_sequences`.
    :param warping_window: see function `segment_repeat_sequences`.
    :param num_templates_to_select: Number of templates to select. If set to None, this is automatically set to
                                    a suitable value.
    :param templates_results_file: Filename for the pickle file in which the processed template results will be
                                   saved. This can be used to avoid processing the templates (which can be time
                                   consuming) repeatedly on multiple runs.
    :param seed: Seed of the random number generator.

    :return results: A dict with the following keys described below:
    - templates_normalized: Selected subset of normalized template sequences per action. A tuple of tuples, where
                            each element of inner tuple is a 2d numpy array with the template sequences.
    - template_labels: Labels of the selected template sequences per action. A tuple of tuples, where each
                       element of the inner tuple is the label.
    - templates_info: Information about the selected subset of template sequences per action. A tuple of tuples,
                      where each element of inner tuple is a namedtuple of type `template_info_tuple`.
    - distance_thresholds: list of upper thresholds on the average DTW distance, one for each action.
    - length_stats: see function `normalize_templates`.
    """
    np.random.seed(seed)
    templates_norm, templates_info, length_stats = normalize_templates(templates, normalize=normalize)

    logger.info("Selecting the best subset of features for each action:")
    feature_mask_per_action = find_best_feature_subset(templates, warping_window=warping_window,
                                                       add_noise_sequences=True, bypass=False)

    logger.info("Calculating the upper threshold on the DTW distance for each action based on the given "
                "template sequences.")
    (distance_thresholds, templates_norm_selected, template_labels_selected, templates_info_selected,
     template_counts) = find_distance_thresholds(
        templates_norm, template_labels, templates_info, feature_mask_per_action, warping_window,
        num_templates_to_select=num_templates_to_select
    )
    results = {
        'templates_normalized': templates_norm_selected,
        'template_labels': template_labels_selected,
        'templates_info': templates_info_selected,
        'template_counts': template_counts,
        'distance_thresholds': distance_thresholds,
        'feature_mask_per_action': feature_mask_per_action,
        'length_stats': length_stats
    }
    if templates_results_file:
        logger.info("Preprocessed template results have been saved to the file: %s", templates_results_file)
        with open(templates_results_file, 'wb') as fp:
            pickle.dump(results, fp)

    return results


def segment_repeat_sequences(data, templates_norm, templates_info, template_counts, feature_mask_per_action,
                             distance_thresholds, length_stats, normalize=True, warping_window=None,
                             length_step=1, offset_step=1, max_overlap=0, approx=False, seed=1234):
    """
    Segment the sequence `data` to closely match the sequences specified in the list `templates_norm`.

    :param data: numpy array of shape (N, d) with float values corresponding to the data sequence.
                 `N` is the number of points in the series and `d` is the dimension of each point in the series.
    :param templates_norm: tuple `[L_1, . . ., L_k]`, where each `L_i` is another tuple `L_i = [s_i1, . . ., s_im]`,
                           and each `s_ij` is a numpy array (of shape (M, d)) corresponding to a template sequence.
                           The template sequences are expected to be normalized.
    :param templates_info: tuple of namedtuples with information about the templates in `templates_norm`.
    :param template_counts: list or numpy array of integer values. First value is the number of actions. The rest of
                            the values are the number of templates correspondsing to each action.
    :param feature_mask_per_action: tuple of length equal to the number of actions, where each item is a boolean
                                    numpy array that acts a feature mask (selected features for a given action).
    :param distance_thresholds: list of float values with length equal to the number of the actions (i.e. length
                                of `templates_norm`). Each value is an upper threshold on the average DTW distance
                                corresponding to templates from a given action.
    :param length_stats: tuple `(min_length, median_length, max_length)` referring to the length of the
                         template subsequences.
    :param normalize: Apply normalization to the templates and the data subsequences, if set to True.
    :param warping_window: Size of the warping window used to constrain the DTW matching path. This is also know as
                           the Sakoe-Chiba band in DTW literature. This can be set to `None` if no warping window
                           constraint is to be applied; else it should be set to a fractional value in (0, 1]. The
                           actual warping window is obtained by multiplying this fraction with the length of the
                           longer sequence. Suppose this window value is `w`, then any point `(i, j)` along the DTW
                           path satisfies `|i - j| <= w`. Setting this to a large value (closer to 1), allows the
                           warping path to be flexible, while setting it to a small value (closer to 0) will constrain
                           the warping path to be closer to the diagonal. Note that a small value can also speed-up
                           the DTW calculation significantly.
    :param length_step: (int) length search is done in increments of this step. Default value is 1.
    :param offset_step: (int) offset search is done in increments of this step. Default value is 1.
    :param max_overlap: (int) maximum allowed overlap between successive segments. Set to 0 for no overlap.
    :param approx: set to True to enable a coarse but faster search over the offsets.
    :param seed: Seed of the random number generator.

    :return: (data_segments, labels)
        - data_segments: list of segmented subsequences, each of which are numpy arrays of shape (m, d) (`m` can be
                         different for each subsequence).
        - labels: list of best-matching template labels for the subsequences, where value `i` corresponds to the
                  templates in position `i - 1` of the input list `templates`. Label value `0` indicates that the
                  corresponding subsequence in `data_segments` could not be matched to any action.
    """
    offset_step_approx = 5 * offset_step
    max_threshold = 1.0001 * max(distance_thresholds)
    np.random.seed(seed)

    logger.info("Length of the input sequence = %d. Dimension of the input sequence = %d.",
                data.shape[0], data.shape[1])
    min_length, median_length, max_length, max_length_deviation = length_stats
    logger.info("Search range for the subsequence length = [%d, %d].", min_length, max_length)

    # Starting from the left end of the sequence, find the subsequences with minimum average DTW distance from
    # the templates. Repeat this iteratively to extract the segments
    data_segments = []
    labels = []
    num_seg = 0
    data_rem = copy.copy(data)
    while data_rem.shape[0] > min_length:
        offset = 0
        info_best = [0, 0, max_threshold, 0]       # offset, sequence_length, min_distance, label
        match = 0
        while (data_rem.shape[0] - offset) > min_length:
            m, d_min, label = search_subsequence(
                data_rem[offset:, :], templates_norm, templates_info, template_counts, feature_mask_per_action,
                min_length, max_length, max_length_deviation, normalize=normalize, warping_window=warping_window,
                length_step=length_step, dist_min_prior=info_best[2]
            )
            if (label > 0) and (d_min <= distance_thresholds[label - 1]):
                if match:
                    if label == info_best[3]:
                        # Same label as the current best match
                        if d_min < info_best[2]:
                            # Lower average DTW distance than the current best match
                            match += 1
                            info_best = [offset, m, d_min, label]

                    else:
                        # Different label from the current best match. In this case, we retain the current best
                        # match and break out of the loop
                        break
                else:
                    # First matching subsequence
                    match = 1
                    info_best = [offset, m, d_min, label]
                    if approx:
                        # If `approx` is set to True, then we backtrack the offset to do a finer search
                        # over offset values
                        offset_new = offset - offset_step_approx + offset_step
                        if offset_new > 0:
                            offset = offset_new
                            continue

            logger.info("Offset = {}, match = {}".format(offset, match))
            if match:
                offset += offset_step
                # Terminate if either of the conditions below is satisfied:
                # 1. Offset exceeds the last index of the best subsequence found so far.
                # 2. If a match has been found for a certain offset value, and a string of increasing values of
                #    the offset does not lead to a better match (lower average DTW), then we break in order to
                #    speed up the search. The choice of 10 is heuristic.
                #
                if (offset - info_best[0]) > 10 or (offset - info_best[0]) >= (info_best[1] - 1):
                    break
            else:
                offset += (offset_step_approx if approx else offset_step)

        if match:
            num_seg += 1
            offset, m, d_min, label = info_best
            if offset > 0:
                # The segment prior to the offset does not match any action. Hence, its label is set to 0
                data_segments.append(data_rem[:offset, :])
                labels.append(0)

            data_segments.append(data_rem[offset:(offset + m), :])
            labels.append(label)
            logger.info("Segment %d: Length of matched segment = %d, offset = %d, matched template label = %d, "
                        "DTW distance = %.6f.", num_seg, m, offset, label, d_min)
            st = max(0, offset + m - max_overlap)
            data_rem = data_rem[st:, :]
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
