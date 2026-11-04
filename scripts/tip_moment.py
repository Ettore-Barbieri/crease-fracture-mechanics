"""Figure 6 (Section 4): bending moment M_yy near the crease tip, infinite plate, lambda = 1 and 10.

D = 1, a = 1, theta_r = 1. M_yy / (k theta_r) on a 241 x 240 grid around the tip, from the kernel
derivatives (paper, Appendix B). Values are clipped to [-1, 1], the range of the colour bar, and
written as an image with the colour map of the figure (blue - white - orange).
Writes figures/tipmap_lam1.png, figures/tipmap_lam10.png (and the grids as .npz in figures/data).
Run time: a few minutes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from _common import NU, FIGDIR, FIGDATA, moment_yy
from creasefm import line_stiffness, glauert_coefficients, theta

D, a = 1.0, 1.0
Lam = line_stiffness(D, NU)
# pgfplots colormap {crease}: blue!60!black, white, orange!80!black
cmap = LinearSegmentedColormap.from_list("crease", [(0, 0, 0.6), (1, 1, 1), (0.8, 0.4, 0)])

gx = np.linspace(a - 0.5, a + 0.5, 241)
gy = np.linspace(-0.5, 0.5, 240)          # even count: no point on the crease line y = 0
X, Y = np.meshgrid(gx, gy, indexing="ij")
for lam in [1, 10]:
    k = lam * Lam / a
    n, A = glauert_coefficients(a, k, D, NU, theta_r=1.0, N=800)
    M = moment_yy(X.ravel(), Y.ravel(), a, lambda t: theta(t, a, n, A), k, D).reshape(X.shape) / k
    np.savez_compressed(os.path.join(FIGDATA, f"tipmap_lam{lam}.npz"), x=gx - a, y=gy, Myy_over_k_theta_r=M)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(M, -1, 1).T, origin="lower", cmap=cmap, vmin=-1, vmax=1,
              interpolation="bicubic", aspect="auto")
    ax.axis("off")
    fig.savefig(os.path.join(FIGDIR, f"tipmap_lam{lam}.png"))
    plt.close(fig)
    print(f"lambda = {lam}: M_yy/(k theta_r) from {M.min():.3f} to {M.max():.3f}", flush=True)
