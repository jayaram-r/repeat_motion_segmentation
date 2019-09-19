import numpy as np
import scipy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def normalize_maxmin(x):
    """
    Perform max-min normalization that scales the values to lie in the range [0, 1].

    :param x: numpy array of shape `(n, d)`. Normalization should be done along the row dimension.
    :return: normalized array of same shape as the input.
    """
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)

    y = np.ones_like(x)
    mask = x_max > x_min
    if np.all(mask):
        y = (x - x_min) / (x_max - x_min)
    else:
        logger.warning("Maximum and minimum values are equal along %d dimensions. "
                       "Setting the normalized values to 1 along these dimension(s).", x.shape[1] - np.sum(mask))
        y[:, mask] = (x[:, mask] - x_min[mask]) / (x_max[mask] - x_min[mask])

    return y


def num_templates_to_sample(n):
    """
    Given `n` templates, what is the number of templates `k` to sample such that the term
    n choose(n - 1, k) is maximized.

    :param n: (int) number of templates.
    :return:
    """
    # Adding a small value to break ties and give preference to the larger `k` in case of ties
    vals = {k: (n * scipy.special.comb(n - 1, k) + 1e-6 * k) for k in range(1, n - 1)}
    k_max = max(vals, key=vals.get)
    v_max = int(np.floor(vals[k_max]))

    return k_max, v_max
