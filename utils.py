import numpy as np
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
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    if x_max > x_min:
        y = (1.0 / (x_max - x_min)) * (x - x_min)
    else:
        logger.warning("Maximum and minimum values are equal. Setting all normalized values to 1.")
        y = np.ones_like(x)

    return y
