import numpy as np


def uniform_simplex(vertices, size=None, rng=None):
    """
    Sample points uniformly from within a simplex defined by vertices.

    Parameters
    ----------
    vertices : array with shape (m, m - 1)
        The vertices of a simplex in (m - 1)-dimensional space.
    size : int, optional
        The sample size.  If size is None, a single point is drawn (an
        array with size (m - 1,).  Otherwise, the output is an array with
        shape (size, m - 1)
    rng : numpy.random.Generator instance, optional
        If not given, a new Generator will used.

    Returns
    -------
    ndarray
        The uniform random samples drawn from within the simplex
        defined by `vertices`.

    """
    vertices = np.array(vertices)
    shp = vertices.shape
    if vertices.ndim != 2 or shp[0] != shp[1] + 1:
        raise ValueError('vertices must be a 2-d array with shape (m, m - 1), '
                         'representing m points in (m - 1)-dimensional space.')
    m = shp[0]

    if rng is None:
        rng = np.random.default_rng()

    if size is None:
        nvars = 1
    else:
        nvars = size

    p = rng.dirichlet(np.ones(m), size=nvars)
    return p @ vertices
