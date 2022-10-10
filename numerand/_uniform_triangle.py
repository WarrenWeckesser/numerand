import numpy as np


def uniform_triangle(vertices, size=None, rng=None):
    """
    Sample points uniformly from within a triangle.

    Parameters
    ----------
    vertices : array with shape (3, n)
        The vertices of a triangle in n-dimensional space.
    size : int, optional
        The sample size.  If size is None, a single point is drawn (an
        array with size (n,).  Otherwise, the output is an array with
        shape (size, n)
    rng : numpy.random.Generator instance, optional
        rng can actually be any object that has the method
        `uniform(low, high, size)`.

    Returns
    -------
    ndarray
        The uniform random samples drawn from within the triangle
        defined by `vertices`.

    """
    vertices = np.atleast_1d(vertices)
    if len(vertices) != 3 or vertices.ndim != 2:
        raise ValueError('vertices must be a 2-d array with shape (3, n), '
                         'representing 3 points in n-dimensional space.')

    if rng is None:
        rng = np.random.default_rng()

    if size is None:
        nvars = 1
    else:
        nvars = size

    s = np.sqrt(rng.uniform(size=(nvars, 1)))
    x = 1 - s
    y = rng.uniform(0, s)
    v01 = vertices[1] - vertices[0]
    v02 = vertices[2] - vertices[0]
    sample = vertices[0] + x*v01 + y*v02
    if size is None:
        sample = sample[0]
    return sample
