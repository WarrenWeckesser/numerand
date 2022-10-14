# To plot the theoretical PDFs of the marginal distributions, this
# script requires scipy.

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from numerand import random_spaced

low = 0
high = 100
delta = 10
n = 5
size = 50000
s = random_spaced(low, high, delta, n=n, size=size)

fig, ax = plt.subplots()

for k in range(s.shape[1]):
    ax.hist(s[:, k], bins=100, density=True, alpha=0.3)
ax.set_title(f"Normalized histograms for components of\n"
             f"random_spaced(low={low}, high={high}, delta={delta}, "
             f"n={n}, size={size})\n"
             "Dashed curves are the marginal PDFs")
ax.grid(alpha=0.2)

# Plot the PDFs of the marginal distributions of each component.
# These are beta distributions.
if (n-1)*delta < high - low:
    for k in range(n):
        left = low + k*delta
        right = high - (n - k - 1)*delta
        dist = beta(k + 1, n - k, loc=left, scale=right - left)
        xx = np.linspace(left, right, 400)
        yy = dist.pdf(xx)
        ax.plot(xx, yy, 'k--', linewidth=1, alpha=0.5)
        if n > 1:
            # Mark the mode with a dot.
            mode0 = k/(n-1)
            mode = (right-left)*mode0 + left
            ax.plot(mode, dist.pdf(mode), 'k.', alpha=0.25)

fig.tight_layout()
plt.show()
