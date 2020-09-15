import numpy as np


def normalize_data(x):
    """Normalize x, x can be a scalar, a vector, or a matrix."""
    # Normalize the mean
    norm_mean = np.mean(x)
    x = x - norm_mean
    # Normalize the variance
    norm_factor = np.sqrt(np.sum(np.square(x)) / x.size)
    assert norm_factor != 0
    return x * 1.0 / norm_factor
