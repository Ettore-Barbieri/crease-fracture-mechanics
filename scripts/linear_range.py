"""Table 4 (Section 8): how far the nonlinear code departs from the linear finite-sheet theory.

Code runs at rest angles 1, 2, 5, 10, 20 deg (h = 0.005) and two runs at h = 0.01 (2 and 5 deg),
for xi = 1 and 10 (data/curvami/CB_n80_t*_xi*_energy.csv and *_theta.csv).
Linear theory: the stored energy is quadratic in theta_r, so it is computed once with theta_r = 1.
Ratios at a = 0.3: stored energy U; G_f from the energy over a = 0.1-0.3; mean fold angle.
"""
import os
import re
import numpy as np
from _common import NU, D_CODE, CODEDATA, FiniteSheet
from creasefm import theta

fs = FiniteSheet(D=D_CODE, nu=NU, P=20, nb=61, nedge=240, ntau=200)
A = np.round(np.arange(0.025, 0.5001, 0.025), 3)
ref = {}
for xi in [1, 10]:
    ref[xi] = [(a,) + tuple(fs.energy(a, xi * D_CODE, 1.0, N=800)) for a in A]   # (a, U, n, A_n)

runs = [(1, 1, ""), (2, 1, ""), (5, 1, ""), (10, 1, ""), (20, 1, ""),
        (1, 10, ""), (2, 10, ""), (5, 10, ""), (10, 10, ""), (20, 10, ""),
        (2, 1, "_h0.01"), (5, 1, "_h0.01")]
print("theta_r, xi, h, eta(a=0.3) | U/U_lin, G_f/G_lin (a = 0.1-0.3), mean theta/lin, at a = 0.3")
for deg, xi, suf in runs:
    name = f"CB_n80_t{deg}_xi{xi}{suf}"
    f = os.path.join(CODEDATA, name + "_energy.csv")
    hdr = open(f).readline()
    h = float(re.search(r"h=([\d.e-]+)", hdr).group(1))
    Dh = float(re.search(r"D=([\d.e-]+)", hdr).group(1))      # D grows like h^3; k = xi D
    th = np.deg2rad(deg)
    d = np.genfromtxt(f, delimiter=",", skip_header=3, names=True)
    a, U = d["a"][1:], d["Pi"][1:]                   # step 0 is the flat sheet
    UL = np.array([r[1] for r in ref[xi]]) * th ** 2 * Dh / D_CODE   # linear U is proportional to D
    i1, i3 = np.argmin(abs(a - 0.1)), np.argmin(abs(a - 0.3))
    G = lambda P: -0.5 * (P[i3] - P[i1]) / (a[i3] - a[i1])
    t = np.loadtxt(os.path.join(CODEDATA, f"{name}_a0.3_theta.csv"), delimiter=",", skiprows=2)
    j = int(np.argmin(abs(A - 0.3)))
    tl = theta(t[:, 0], 0.3, ref[xi][j][2], ref[xi][j][3]) * th
    print(f"{deg:3d} deg  xi={xi:3d}  h={h}  eta={th * 0.3 / h:5.2f} | "
          f"{U[i3] / UL[i3]:.3f}  {G(U) / G(UL):.2f}  {t[:, 1].sum() / tl.sum():.3f}")
