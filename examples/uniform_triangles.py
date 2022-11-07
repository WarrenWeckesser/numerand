import numpy as np
import matplotlib.pyplot as plt
from numerand import uniform_simplices


rng = np.random.default_rng(121263137472525314065)
vertices = np.array([[0, 0], [1, 1], [2, 1], [2, 0], [4, 0], [0, 2], [2, 2]])
simplices = np.array([[0, 5, 6], [2, 1, 6], [3, 6, 4]])

pts = uniform_simplices(vertices, simplices, size=5000, rng=rng)

fig, ax = plt.subplots()
ax.plot(pts[:, 0], pts[:, 1], '.',
        markersize=3, markeredgewidth=0)
ax.plot(vertices[:, 0], vertices[:, 1], 'ko',
        alpha=0.75, markersize=4)

ax.grid(True, alpha=0.5)
ax.axis('equal')
ax.set_title('Samples drawn uniformly from the union of triangles\n')
plt.show()
