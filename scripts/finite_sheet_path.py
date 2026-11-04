"""Figures 10 (stored energy and G_f along the path a = 0 ... W) and 12 (size effect), Section 7.

Finite square sheet |x|, |y| <= W = 1 with free edges, elastic hinge, rest angle theta_r = 1, D = 1.
The Ritz basis grows near the edges, where the correction becomes steep.
Writes figures/data/path_xi*.dat, pathU_xi*.dat, pathU_end_xi*.dat, size_W*.dat.
Run time: about 30-60 minutes.
"""
import numpy as np
from _common import save_columns, NU, FiniteSheet
from creasefm import line_stiffness, fold_energy_release_rate

D, th0 = 1.0, 1.0
Lam = line_stiffness(D, NU)
fs0 = FiniteSheet(D=D, nu=NU, P=20, nb=61, nedge=240, ntau=200)
fsA = FiniteSheet(D=D, nu=NU, P=28, nb=121, nedge=400, ntau=300)
fsB = FiniteSheet(D=D, nu=NU, P=44, nb=201, nedge=800, ntau=600)


def sheet(a):
    return fs0 if a <= 0.7 else (fsA if a <= 0.9 else fsB)


def G_and_U(a, k):
    """G_f = -1/2 dU/da by central difference, and U, in the finite sheet."""
    fs = sheet(a)
    da = 1e-3 * min(1, (1 - a) / 0.05)
    Up = fs.energy(a + da, k, th0, N=800)[0]
    Um = fs.energy(a - da, k, th0, N=800)[0]
    U0 = fs.energy(a, k, th0, N=800)[0]
    return -0.5 * (Up - Um) / (2 * da), U0


# Figure 10: path for xi = k W / D = 1, 10, 100
A_path = np.r_[np.arange(0.05, 0.9, 0.05), 0.9, 0.925, 0.95, 0.97]
for xi in [1, 10, 100]:
    k = xi * D
    rows = []
    for a in A_path:
        g, U = G_and_U(a, k)
        gi = fold_energy_release_rate(a, k, D, NU, th0, N=800)["Gf"]
        rows.append((a, U, g, gi))
        print(f"path xi={xi} a={a:.3f} G/(k/2)={g / (0.5 * k):+.4f} inf={gi / (0.5 * k):+.4f} U/Lam={U / Lam:.4f}", flush=True)
    R = np.array(rows)
    save_columns(f"path_xi{xi}.dat", [R[:, 0], R[:, 1] / Lam, R[:, 2] / Lam, R[:, 3] / Lam], "a U Gf Ginf")
    # stored energy: U = 0 with no crease, a few short creases, the path, and U = 0 at a = W (dotted)
    small = [(0.0, 0.0)] + [(a, fs0.energy(a, k, th0, N=800)[0] / Lam) for a in [0.005, 0.01, 0.02, 0.03, 0.04]]
    S = np.array(small + [(r[0], r[1] / Lam) for r in rows])
    save_columns(f"pathU_xi{xi}.dat", [S[:, 0], S[:, 1]], "a U")
    save_columns(f"pathU_end_xi{xi}.dat", [[A_path[-1], 1.0], [R[-1, 1] / Lam, 0.0]], "a U")

# Figure 12: size effect, W / l_k = 0.3, 3, 30 (l_k = Lam / k)
for lamW, tag in [(0.3, "0p3"), (3, "3"), (30, "30")]:
    k = lamW * Lam
    rows = []
    for a in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        g, U = G_and_U(a, k)
        rows.append((a * lamW, g / (0.5 * k)))
        print(f"size W/lk={lamW} a={a}: G/(k/2)={g / (0.5 * k):+.4f}", flush=True)
    R = np.array(rows)
    save_columns(f"size_W{tag}.dat", [R[:, 0], R[:, 1]], "lam G")
