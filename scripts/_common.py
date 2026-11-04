"""Shared paths, constants and helpers for the figure and table scripts."""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from creasefm import line_stiffness, theta  # noqa: E402
from creasefm.finite_sheet import FiniteSheet, _DER  # noqa: E402

FIGDATA = os.path.join(ROOT, "figures", "data")      # data read by the pgfplots figures
FIGDIR = os.path.join(ROOT, "figures")
CODEDATA = os.path.join(ROOT, "data", "curvami")      # processed output of the meshfree code
os.makedirs(FIGDATA, exist_ok=True)

NU = 0.3
D_CODE = 1.0989010989            # bending stiffness of the code runs (E = 9.6e7, h = 0.005, nu = 0.3)
THETA_CODE = np.deg2rad(0.5)     # rest angle of the code runs in Section 8


def save_columns(name, cols, header, fmt="%.6g"):
    """Write columns with a one-line header, as read by pgfplots."""
    np.savetxt(os.path.join(FIGDATA, name), np.column_stack(cols), header=header,
               comments="", fmt=fmt)


def write_lines(name, X, Y, Z, rows):
    """Write the scan lines X[i, :], Y[i, :], Z[i, :] for i in rows, separated by blank lines."""
    with open(os.path.join(FIGDATA, name), "w") as f:
        f.write("x y w\n")
        for i in rows:
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]:.4f} {Y[i, j]:.4f} {Z[i, j]:.6g}\n")
            f.write("\n")


# Gauss rule on the crease, for fields of the crease by direct quadrature
_TQ, _WQ = np.polynomial.legendre.leggauss(3000)


def w_infinite(X, Y, a, th, nu=NU):
    """Infinite-plate deflection of a crease with fold angle th (sampled at the Gauss points a*_TQ),
    w = int theta(t) K(x - t, y) dt (paper, Section 3.3). Defined up to a rigid-body motion."""
    W = np.zeros_like(X, dtype=float)
    for t, ww, tt in zip(_TQ * a, _WQ * a, th):
        s = X - t
        r2 = s * s + Y * Y
        W += ww * tt * ((1 + nu) * np.log(r2) + (1 + nu) + 2 * (Y * Y + nu * s * s) / r2) / (8 * np.pi)
    return W


def theta_at_gauss(a, n, A):
    """Fold angle at the Gauss points used by w_infinite."""
    return theta(_TQ * a, a, n, A)


def rigid_plane(x, y, w):
    """Least-squares plane c0 + c1 x + c2 y fitted to w (a rigid-body motion of a free sheet)."""
    A3 = np.c_[np.ones_like(x), x, y]
    return np.linalg.lstsq(A3, w, rcond=None)[0]


def moment_yy(X, Y, a, th_fun, k, D, nu=NU, nquad=8000, chunk=400, ws=None):
    """M_yy of the crease field at points (X, Y), by quadrature of the kernel derivatives.
    th_fun(t): fold angle on the crease. ws: optional (basis derivatives function, coefficients)
    of a smooth correction w_c to add (finite sheet)."""
    tg, tw = np.polynomial.legendre.leggauss(nquad)
    t = a * tg
    tw = a * tw
    th = th_fun(t)
    acc = {}
    for key in [(2, 0), (0, 2)]:
        f = _DER[key]
        s = np.zeros_like(X, dtype=float)
        for i in range(0, len(t), chunk):
            tt = t[i:i + chunk]
            s += (f(X[:, None] - tt[None, :], Y[:, None] * np.ones((1, len(tt))), nu)
                  * (tw[i:i + chunk] * th[i:i + chunk])[None, :]).sum(1)
        acc[key] = s
    wyy, wxx = acc[(0, 2)], acc[(2, 0)]
    if ws is not None:
        basis, c = ws
        d = basis(X, Y)
        wyy = wyy + d["yy"] @ c
        wxx = wxx + d["xx"] @ c
    return -D * (wyy + nu * wxx)
