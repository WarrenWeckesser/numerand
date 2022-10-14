import numpy as np
import matplotlib.pyplot as plt
from numerand import uniform_simplex


rng = np.random.default_rng(3486780953123987)

vertices = np.array([[0, 0, 0], [4, 0, 4], [2, 3, 4], [5, 2, 1]])

pts = uniform_simplex(vertices, size=2000, rng=rng)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='.', alpha=0.2)

for j in range(len(vertices) - 1):
    for k in range(j + 1, len(vertices)):
        ax.plot([vertices[j, 0], vertices[k, 0]],
                [vertices[j, 1], vertices[k, 1]],
                [vertices[j, 2], vertices[k, 2]], 'k--')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(True, alpha=0.5)
ax.axis('equal')
ax.set_title('Samples drawn uniformly from the tetrahedron\n'
             f'defined by the vertices {vertices.tolist()}')
plt.show()
