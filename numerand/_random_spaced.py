# I originally wrote this for the stackoverflow question
# https://stackoverflow.com/questions/53565205/
#     how-to-generate-random-numbers-with-each-random-number-having-a-difference-of-at

import numpy as np


def random_spaced(low, high, delta, n, size=None, rng=None):
    """
    Choose n random values between low and high, with minimum spacing delta.

    If size is None, one sample is returned.
    Set size=m (an integer) to return m samples.

    The values in each sample returned by random_spaced are in increasing
    order.

    The marginal distribution of each component is a beta distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from numerand import random_spaced

    >>> rng = np.random.default_rng(0x1ce1cebab1e)
    >>> x = random_spaced(0, 1, 0.2, n=3, size=10, rng=rng)
    >>> x
    array([[0.02385442, 0.3250959 , 0.81466327],
           [0.02239674, 0.58901262, 0.78980063],
           [0.33271347, 0.66223986, 0.87430722],
           [0.41440712, 0.74215534, 0.94302544],
           [0.11192088, 0.3138193 , 0.82968734],
           [0.1983123 , 0.51938522, 0.91260255],
           [0.0727319 , 0.47950355, 0.75712988],
           [0.38420333, 0.67328081, 0.98850325],
           [0.01257311, 0.36606718, 0.74807102],
           [0.32475111, 0.65489123, 0.99844347]])

    Check that the spacing in each sample is at least 0.2.

    >>> np.diff(x, axis=1).min()
    0.20078801655183665

    """
    if rng is None:
        rng = np.random.default_rng()

    space = high - low - (n-1)*delta
    if space < 0:
        raise ValueError(f"It is not possible to select {n} points from the "
                         f"interval {[low, high]} with minimum spacing "
                         f"{delta}.")

    if size is None:
        u = rng.random(n)
    else:
        u = rng.random((size, n))
    x = space * np.sort(u, axis=-1)
    return low + x + delta * np.arange(n)
