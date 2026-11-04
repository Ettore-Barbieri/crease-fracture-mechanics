"""Figure 7 (fold angle and fold energy release rate against the hinge number) and Table 1 (Section 5).

Infinite plate, elastic hinge with rest angle theta_r, no load. Units: D = 1, a = 1, theta_r = 1.
Writes figures/data/rest_theta.dat and rest_Gf.dat, and prints Table 1 and the two limits (Section 5.3).
"""
import numpy as np
from _common import save_columns, NU
from creasefm import line_stiffness, glauert_coefficients, theta, stored_energy, fold_energy_release_rate

D, a = 1.0, 1.0
Lam = line_stiffness(D, NU)

# (a) fold angle along the crease, theta / theta_r, for four hinge numbers
x = np.linspace(-1, 1, 201)
cols = [x]
for lam in [0.1, 1, 10, 100]:
    n, A = glauert_coefficients(a, lam * Lam / a, D, NU, theta_r=1.0, N=600)
    cols.append(theta(x, a, n, A))
save_columns("rest_theta.dat", cols, "x l0p1 l1 l10 l100")

# (b) G_f / (k theta_r^2 / 2) against lambda
lams = np.logspace(-2, 2, 49)
G = [fold_energy_release_rate(a, l * Lam / a, D, NU, 1.0, N=800)["Gf"] / (0.5 * l * Lam / a) for l in lams]
save_columns("rest_Gf.dat", [lams, G], "lam G")

# Table 1
print("Table 1: lambda, theta(0)/theta_r, g, G_f/(k theta_r^2/2), G_f a/(Lam theta_r^2)")
for lam in [0.01, 0.1, 0.3, 1, 3, 10, 100]:
    k = lam * Lam / a
    n, A = glauert_coefficients(a, k, D, NU, theta_r=1.0, N=800)
    g = stored_energy(a, k, D, NU, 1.0, N=800) / Lam
    Gf = fold_energy_release_rate(a, k, D, NU, 1.0, N=800)["Gf"]
    print(f"{lam:6g}  {theta(0.0, a, n, A)[0]:.4f}  {g:.4f}  {Gf / (0.5 * k):+.4f}  {Gf * a / Lam:+.4f}")

# Limits, Eq. (limits): short crease G_f/(k/2) = -(1 - pi lam); long crease G_f a / Lam = -1/(4 pi)
print("lambda = 0.01: exact", f"{np.interp(0.01, lams, G):.4f}", " short-crease limit", f"{-(1 - np.pi * 0.01):.4f}")
Gf100 = fold_energy_release_rate(a, 100 * Lam / a, D, NU, 1.0, N=800)["Gf"]
print("lambda = 100:  exact", f"{Gf100 * a / Lam:.4f}", " long-crease limit", f"{-1 / (4 * np.pi):.4f}")
