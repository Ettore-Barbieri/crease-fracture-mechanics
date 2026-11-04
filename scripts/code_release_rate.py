"""Figure 15 and Table 3 (Section 8): fold energy release rate, finite-sheet theory against the code.

Theory: G_f = -1/2 dU/da in the finite sheet (central difference), and in the infinite plate.
Code: domain integral on rings around both tips, analytic weight q
(data/curvami/CB_n80_t0.5_GD_table.csv, form "_s"), and the energy route (form "energy").
All in units of k theta_r^2 / 2. Writes figures/data/fsGf_xi*.dat and fsGf_code_xi*.dat.
"""
import os
import csv
import numpy as np
from _common import NU, D_CODE, THETA_CODE, CODEDATA, FiniteSheet, save_columns
from creasefm import fold_energy_release_rate

fs = FiniteSheet(D=D_CODE, nu=NU, P=20)
rows = list(csv.reader(l for l in open(os.path.join(CODEDATA, "CB_n80_t0.5_GD_table.csv"))
                       if not l.startswith("#")))[1:]      # xi, a, tip, rho1, rho2, form, G_D, ...


def G_finite(a, k, da=1e-3):
    U = lambda aa: fs.energy(aa, k, THETA_CODE, N=600)[0]
    return -0.5 * (U(a + da) - U(a - da)) / (2 * da)


A_code = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
table = {}
for xi in [1, 10, 100]:
    k = xi * D_CODE
    u = 0.5 * k * THETA_CODE ** 2
    # theory curves
    aa = np.linspace(0.04, 0.46, 22)
    gf = [G_finite(a, k) for a in aa]
    gi = [fold_energy_release_rate(a, k, D_CODE, NU, THETA_CODE, N=800)["Gf"] for a in aa]
    save_columns(f"fsGf_xi{xi}.dat", [aa, np.array(gf) / u, np.array(gi) / u], "a Gfs Ginf")
    # code: mean and range over rings and tips
    m, lo, hi = [], [], []
    for a in A_code:
        s = np.array([float(r[6]) for r in rows
                      if r[0] == str(xi) and abs(float(r[1]) - a) < 1e-9 and r[5] == "_s"]) / u
        m.append(s.mean()); lo.append(s.mean() - s.min()); hi.append(s.max() - s.mean())
        gfa = G_finite(a, k)
        gia = fold_energy_release_rate(a, k, D_CODE, NU, THETA_CODE, N=800)["Gf"]
        en = [float(r[6]) for r in rows if r[0] == str(xi) and abs(float(r[1]) - a) < 1e-9 and r[5] == "energy"][0]
        table[(xi, a)] = (gfa / gia, s.mean() * u / gfa, en / gfa)
    save_columns(f"fsGf_code_xi{xi}.dat", [A_code, m, lo, hi], "a m lo hi")

print("Table 3: a/W | for xi = 1, 10, 100: finite/infinite, code (domain, mean over rings)/finite")
for a in [0.25, 0.3, 0.35, 0.4]:
    print(f"{a:.2f} " + "  ".join(f"{table[(xi, a)][0]:.3f} {table[(xi, a)][1]:.2f}" for xi in [1, 10, 100]))
print("Energy route, code/finite, xi = 1:", " ".join(f"{table[(1, a)][2]:.2f}" for a in A_code))
