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

    p = rng.dirichlet(np.ones(m), size=size)
    return p @ vertices


def uniform_simplices(points, simplices, size=None, rng=None):
    """
    Generate samples uniformly in the union of a collection of simplices.

    The simplices are assumed to be disjoint except for possibly shared
    vertices, edges, faces, etc.  That is, the measure of the intersection
    of any pair of vertices must be 0.

    Parameters
    ----------
    points : array_like, shape (m, dim)
        Points that are vertices of the simplices
    simplices : array_like, shape (n, dim+1)
        Each row of `simplices` gives the indices into `points` that
        are the vertices of the simplex.
    size : int, optional
        The sample size.
    rng : numpy.random.Generator instance, optional
        If not given, a new Generator will be used.

    Returns
    -------
    ndarray
        The uniform random samples drawn from the union of the simplices.

    """ 
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError('points must be a 2-d array')
    m, dim = points.shape
    simplices = np.asarray(simplices)
    if simplices.ndim != 2:
        raise ValueError('simplices must be a 2-d array')
    n, k = simplices.shape
    if k != dim + 1:
        raise ValueError(f'Each simplex must have {dim+1} indices into the points array')

    if rng is None:
        rng = np.random.default_rng()

    volumes = []
    df = 1
    for i in range(1, dim+1):
        df *= i
    for simplex in simplices:
        m = np.ones((k, k))
        m[:, :-1] = points[simplex]
        volumes.append(np.abs(np.linalg.det(m))/df)
    volumes = np.array(volumes)
    p = volumes/volumes.sum()
    counts = rng.multinomial(1 if size is None else size, p)
    samples = []
    for i, count in enumerate(counts):
        samples.append(uniform_simplex(points[simplices[i]], size=count, rng=rng))
    return np.concatenate(samples)
