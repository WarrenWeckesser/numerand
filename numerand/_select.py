
import operator
import numpy as np


# TO DO:  Add an out parameter.

def select(items, *, nsample=None, p=None, size=None, rng=None):
    """
    Select random samples from `items`.

    The function randomly selects `nsample` items from `items` without
    replacement. ``nsample=None`` implies a single item is selected.
    Sampling *with* replacement is accomplished by leaving ``nsample=None``,
    and setting `size` to the size of the sample to be generated.  See the
    Parameters and Notes sections for full details.

    Parameters
    ----------
    items : sequence
        The collection of items from which the selection is made.
    nsample : int, optional
        If `nsample` is ``None``, a single item is selected. If it is an
        integer, `nsamples` must between 0 and ``len(items)`` (inclusive),
        and a sequence of length `nsample` is selected by drawing `nsample`
        items without replacement.  See the notes below for more details.
    p : array-like of floats, same length as `items`, optional
        Probabilities of the items.  If this argument is not given, the
        elements in `items` are assumed to have equal probability.
    size : int or tuple of ints, optional
        Number of variates to generate. Each "variate" is either a scalar
        (if `nsample` is ``None``) or a 1-d array of length `nsample`.
    rng : numpy.random.Generator instance, optional
        If not given, ``numpy.random.default_rng()`` is used.

    Returns
    -------
    selection : ndarray
        When both `nsample` and `size` are integers, the shape of the
        return value is ``(size, nsample)``.  When `size` is a tuple, the
        shape is ``size + (nsample,)``.

    Notes
    -----
    ``size=None`` means "generate a single selection".

    If `size` is None, the result is equivalent to

        choice(items, size=nsample, replace=False)

    ``nsample=None`` means draw one sample.

    If `nsample` is None, the functon acts (almost) like nsample=1 (see
    below for more information), and the result is equivalent to

        choice(items, size=size)

    In effect, it does choice with replacement.  The case ``nsample=None``
    can be interpreted as each sample is a scalar, and ``nsample=k``
    means each sample is a sequence with length ``k``.

    If `nsample` is not None, it must be a nonnegative integer with
    ``0 <= nsample <= len(items)``.

    If `size` is not None, it must be an integer or a tuple of integers.
    When `size` is an integer, it is treated as the tuple ``(size,)``.

    When both `nsample` and `size` are not None, the result
    has shape ``size + (nsample,)``.

    Examples
    --------
    Make 6 choices with replacement from [10, 20, 30, 40].  This is
    equivalent to "Make 1 choice without replacement from [10, 20, 30, 40];
    do it six times."

    >>> select([10, 20, 30, 40], size=6)
    array([20, 20, 40, 10, 40, 30])

    Choose two items from [10, 20, 30, 40] without replacement.  Do it six
    times.

    >>> select([10, 20, 30, 40], nsample=2, size=6)
    array([[40, 10],
           [20, 30],
           [10, 40],
           [30, 10],
           [10, 30],
           [10, 20]])

    When `nsample` is an integer, there is always an axis at the end of the
    result with length `nsample`, even when ``nsample=1``.  For example, the
    shape of the array returned in the following call is (2, 3, 1)

    >>> select([10, 20, 30, 40], nsample=1, size=(2, 3))
    array([[[10],
            [30],
            [20]],

           [[10],
            [40],
            [20]]])

    When `nsample` is None, it acts like ``nsample=1``, but the trivial
    dimension is not included.  The shape of the array returned in the
    following call is (2, 3).

    >>> select([10, 20, 30, 40], size=(2, 3))
    array([[20, 40, 30],
           [30, 20, 40]])

    """
    # This implementation is a proof of concept, and provides a demonstration
    # of the `select` API.  Efficiency was not considered.  The actual
    # implementation will probably use Cython or C.

    if rng is None:
        rng = np.random.default_rng()

    if nsample is None:
        return rng.choice(items, size=size, p=p)

    # Ensure that nsample is a nonnegative integer.
    nsample = operator.index(nsample)
    if nsample < 0:
        raise ValueError('nsample must be nonnegative.')
    if nsample > len(items):
        raise ValueError(f'nsample ({nsample}) must not exceed '
                         F'len(items) ({len(items)})')

    if size is None:
        size = ()
    elif np.isscalar(size):
        size = (size,)

    # FIXME: check that this product doesn't overflow.
    if np.prod(size, dtype=np.intp) > 0:

        def func(_):
            return rng.choice(items, size=nsample, p=p, replace=False)

        tmp = np.empty(size + (0,))
        result = np.apply_along_axis(func, -1, tmp)
    else:
        # There is a 0 in `size`.
        result = np.empty(size + (nsample,), dtype=np.asarray(items).dtype)
    return result
