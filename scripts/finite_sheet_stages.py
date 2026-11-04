"""Figure 11 (Section 7): deflection of the square sheet as the centre crease is extended (finite-sheet theory, xi = 10).

W = 1, D = 1, k = 10, theta_r = 1. The deflection is evaluated on a 41 x 41 grid, the rigid-body plane
is removed, and the grid is written as scan lines: sections across the crease at fixed x
(on the crease: *_xin.dat; beyond its tips: *_xout.dat) and lines along it at fixed y (*_y.dat).
Run time: about 10-20 minutes.
"""
import numpy as np
from _common import NU, FiniteSheet, write_lines, w_infinite, theta_at_gauss, rigid_plane

D, k = 1.0, 10.0
fs0 = FiniteSheet(D=D, nu=NU, P=20, nb=61, nedge=240, ntau=200)
fsA = FiniteSheet(D=D, nu=NU, P=28, nb=121, nedge=400, ntau=300)
fsB = FiniteSheet(D=D, nu=NU, P=44, nb=201, nedge=800, ntau=600)

g = np.linspace(-1, 1, 41)
X, Y = np.meshgrid(g, g, indexing="ij")       # X[i, :] is a section at fixed x
x, y = X.ravel(), Y.ravel()
ys = np.where(y == 0, 1e-9, y)                # evaluate the crease line on its upper side

for a in [0.25, 0.5, 0.75, 0.95]:
    fs = fs0 if a <= 0.7 else (fsA if a <= 0.9 else fsB)
    n, A = fs.solve(a, k, 1.0, N=800)
    w = w_infinite(x, ys, a, theta_at_gauss(a, n, A)) + fs.w_correction(x, ys)
    c = rigid_plane(x, y, w)
    w = (w - np.c_[np.ones_like(x), x, y] @ c).reshape(X.shape)
    tag = f"stage_a{int(round(a * 100))}"
    xrows = range(0, 41, 2)
    write_lines(tag + "_xin.dat", X, Y, w, [i for i in xrows if abs(g[i]) < a])
    write_lines(tag + "_xout.dat", X, Y, w, [i for i in xrows if abs(g[i]) >= a])
    write_lines(tag + "_y.dat", X.T, Y.T, w.T, range(0, 41, 4))
    print(f"a = {a}: w/(theta_r W) from {w.min():.4f} to {w.max():.4f}", flush=True)
