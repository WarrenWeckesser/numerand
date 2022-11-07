# This example requires SciPy.

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from numerand import uniform_simplices


rng = np.random.default_rng(121263137472525314065)

# Generate 9 vertices on the unit sphere.
z = rng.normal(size=(9, 3))
vertices = z / np.linalg.norm(z, axis=1, keepdims=True)

# Compute the Delaunay triangulation, and use the result as the
# input to uniform_simplices.
dt = Delaunay(vertices)
pts = uniform_simplices(dt.points, dt.simplices, size=2500, rng=rng)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='.', alpha=0.2)

ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k.')
for idx in dt.convex_hull:
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        seg = vertices[[idx[i], idx[j]]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], 'k-', alpha=0.2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True, alpha=0.5)
ax.axis('equal')
ax.set_title('Samples drawn uniformly from an irregular polyhedron\n')
plt.show()
