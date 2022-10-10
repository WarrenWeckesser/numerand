import numpy as np
import matplotlib.pyplot as plt
from numerand import uniform_simplex


rng = np.random.default_rng(3486780953123987)
vertices = np.array([[0, 1], [4, 2], [10, -2]])

pts = uniform_simplex(vertices, size=4000, rng=rng)

fig, ax = plt.subplots()
ax.plot(pts[:, 0], pts[:, 1], '.',
        markersize=3, markeredgewidth=0)
ax.plot(vertices[:, 0], vertices[:, 1], 'ko',
        alpha=0.75, markersize=4)

ax.grid(True, alpha=0.5)
ax.axis('equal')
ax.set_title('Samples drawn uniformly from the triangle\n'
             f'defined by the vertices {vertices.tolist()}')
plt.show()
