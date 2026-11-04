"""Figures 13 and 14 (Section 8): deflection from the finite-sheet theory and from the meshfree code (xi = 10).

Code runs: W = 1, D = 1.0989010989, h = 0.005, nu = 0.3, theta_r = 0.5 deg, k = 10 D, on 80 x 80 cells
(data/curvami/CB_n80_t0.5_xi10_a*_field.csv, values at the cell-centre Gauss points).
The theory is evaluated at the same points. From each field the rigid-body plane is removed,
and w is divided by theta_r.
Figure 13: every second point, written as scan lines (cmp_{theory,code}_a*_{xin,xout,y}.dat).
Figure 14: sections across the crease at x = 0 (sec_{theory,code}_a*.dat).
"""
import os
import numpy as np
from _common import (NU, D_CODE, THETA_CODE, CODEDATA, FiniteSheet, write_lines, save_columns,
                     w_infinite, theta_at_gauss, rigid_plane)
from creasefm import theta

k = 10 * D_CODE
fs = FiniteSheet(D=D_CODE, nu=NU, P=20)

for a in [0.1, 0.3, 0.5]:
    d = np.loadtxt(os.path.join(CODEDATA, f"CB_n80_t0.5_xi10_a{a}_field.csv"), delimiter=",", skiprows=2)
    x, y, w = d[:, 0], d[:, 1], d[:, 4]
    o = np.lexsort((np.round(y, 6), np.round(x, 6)))          # x outer, y inner
    x, y, w = x[o], y[o], w[o]
    n, A = fs.solve(a, k, THETA_CODE, N=600)
    th = theta_at_gauss(a, n, A)
    wt = w_infinite(x, y, a, th) + fs.w_correction(x, y)
    A3 = np.c_[np.ones_like(x), x, y]
    cc, ct = rigid_plane(x, y, w), rigid_plane(x, y, wt)
    WC = ((w - A3 @ cc) / THETA_CODE).reshape(80, 80)
    WT = ((wt - A3 @ ct) / THETA_CODE).reshape(80, 80)
    XX, YY = x.reshape(80, 80), y.reshape(80, 80)

    # Figure 13: every second point; sections every second column, lines along the crease every fourth row
    ix = np.r_[np.arange(1, 40, 2), np.arange(40, 80, 2)]
    S = np.ix_(ix, ix)
    Xc, Yc = XX[S], YY[S]
    xs = Xc[:, 0]
    tag = f"a{int(round(a * 10))}"
    xr = range(0, 40, 2)
    yr = list(range(0, 40, 4)) + [39]
    for name, Wf in [("code", WC[S]), ("theory", WT[S])]:
        write_lines(f"cmp_{name}_{tag}_xin.dat", Xc, Yc, Wf, [i for i in xr if abs(xs[i]) < a])
        write_lines(f"cmp_{name}_{tag}_xout.dat", Xc, Yc, Wf, [i for i in xr if abs(xs[i]) >= a])
        write_lines(f"cmp_{name}_{tag}_y.dat", Xc.T, Yc.T, Wf.T, yr)

    # Figure 14: section at x = 0 (code: mean of the two columns next to x = 0)
    save_columns(f"sec_code_{tag}.dat", [YY[0, :], 0.5 * (WC[39, :] + WC[40, :])], "y w")
    yy = np.r_[np.linspace(-1, -1e-9, 200), np.linspace(1e-9, 1, 200)]
    xx = 0 * yy
    wtt = (w_infinite(xx, yy, a, th) + fs.w_correction(xx, yy) - (ct[0] + ct[2] * yy)) / THETA_CODE
    save_columns(f"sec_theory_{tag}.dat", [yy, wtt], "y w")

    diff = np.abs(WC - WT).max() / (WT.max() - WT.min())
    print(f"a = {a}: theta(0)/theta_r = {theta(0.0, a, n, A)[0] / THETA_CODE:.4f}, "
          f"largest difference = {100 * diff:.1f} % of the range of w", flush=True)
