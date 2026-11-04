"""
Centre crease with an elastic hinge in an infinite, isotropic, linear Kirchhoff plate.
Paper: E. Barbieri, "Creases in the light of fracture mechanics", Sections 2-5 and Appendix A.

Crease on |x| < a, y = 0. Hinge law on the crease: M_yy = -k (theta - theta_r).
Optional remote moment M_yy = M0. Solved with Glauert's sine series:
    x = a cos(tau),  theta = sum_n A_n sin(n tau).

Conventions (as in the paper):
    M_ab = -D [(1-nu) w_,ab + nu Lap(w) delta_ab]      (Timoshenko sign convention)
    theta(x) = w_,y(x, 0+) - w_,y(x, 0-)               (fold angle = jump in normal slope)
    Lam = D (1-nu)(3+nu) / 2                            (line stiffness)
    lam = k a / Lam                                     (dimensionless hinge stiffness)

All functions take plain floats / numpy arrays. Units: any consistent set.
"""
import numpy as np


def line_stiffness(D, nu):
    """Lam = D (1 - nu)(3 + nu) / 2."""
    return D * (1.0 - nu) * (3.0 + nu) / 2.0


def glauert_coefficients(a, k, D, nu, theta_r=0.0, M0=0.0, N=400):
    """Coefficients A_1..A_N of theta = sum A_n sin(n tau), x = a cos(tau).

    Galerkin system (paper, Appendix A):
      (Lam/2a)(pi/2) m A_m + k sum_n I_mn A_n = (k theta_r - M0)(pi/2) delta_m1,
      I_mn = 1/2 [c(n-m) - c(n+m)],  c(p) = 2/(1-p^2) for p even, 0 for p odd.
    Use N >= 400 for lam >= 10, N >= 800 for lam >= 100.
    """
    Lam = line_stiffness(D, nu)
    n = np.arange(1, N + 1)
    Mi, Ni = np.meshgrid(n, n, indexing="ij")

    def c(P):
        P = P.astype(float)
        out = np.zeros_like(P)
        even = (P.astype(int) % 2 == 0)
        out[even] = 2.0 / (1.0 - P[even] ** 2)
        return out

    A = k * 0.5 * (c(np.abs(Ni - Mi)) - c(Ni + Mi))
    A[np.diag_indices(N)] += Lam / (2.0 * a) * n * np.pi / 2.0
    b = np.zeros(N)
    b[0] = (k * theta_r - M0) * np.pi / 2.0
    return n, np.linalg.solve(A, b)


def theta(x, a, n, An):
    """Fold angle theta(x) on |x| <= a from the series."""
    tau = np.arccos(np.clip(np.asarray(x, float) / a, -1.0, 1.0))
    return np.sin(np.outer(np.atleast_1d(tau), n)) @ An


def moment_on_line(x, a, n, An, D, nu, k=0.0, theta_r=0.0, M0=0.0):
    """M_yy(x, 0) on the line of the crease.

    On the crease (|x| < a): hinge law, M_yy = -k (theta - theta_r).
    Outside (|x| > a): M0 + plate field of the crease, from
      int_0^pi cos(n phi)/(xi - cos phi) dphi = pi r^n / sqrt(xi^2 - 1),  r = |xi| - sqrt(xi^2 - 1),
    giving M_yy = M0 - sgn(xi)^(n+1)... summed below (odd and even n handled by the sign factor).
    """
    Lam = line_stiffness(D, nu)
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    inside = np.abs(x) < a
    if inside.any():
        out[inside] = -k * (theta(x[inside], a, n, An) - theta_r)
    xo = x[~inside]
    if xo.size:
        xi = xo / a
        s = np.sign(xi)
        rt = np.sqrt(xi ** 2 - 1.0)
        r = np.abs(xi) - rt
        # PV int_{-a}^{a} theta'(t)/(t - x) dt = -(1/a) sum n A_n J_n,
        # J_n = int_0^pi cos(n phi)/(cos phi - xi) dphi = -s^(n+1) pi r^n / rt
        J = -(s[:, None] ** (n[None, :] + 1)) * np.pi * r[:, None] ** n[None, :] / rt[:, None]
        cauchy = -(1.0 / a) * (J * (n * An)[None, :]).sum(1)
        out[~inside] = M0 - Lam / (2.0 * np.pi) * cauchy
    return out


def moment_intensity(a, n, An, D, nu, end=+1):
    """K_M = lim sqrt(2 pi r) M_yy ahead of the tip x = end*a (paper, Section 4)."""
    Lam = line_stiffness(D, nu)
    sgn = np.ones_like(n) if end > 0 else (-1.0) ** (n + 1)
    return -(Lam / 2.0) * np.sqrt(np.pi / a) * np.sum(sgn * n * An)


def stored_energy_rest_angle(a, k, D, nu, theta_r, N=800):
    """Stored energy U = plate + hinge energy, no load (paper, Section 5.1):
    U = 1/2 k theta_r int (theta_r - theta) dx, with int theta dx = a (pi/2) A_1."""
    n, An = glauert_coefficients(a, k, D, nu, theta_r=theta_r, M0=0.0, N=N)
    return 0.5 * k * theta_r * (2.0 * a * theta_r - a * np.pi / 2.0 * An[0])


def hinge_energy(a, k, D, nu, theta_r, N=800, nquad=2000):
    """Hinge energy 1/2 k int (theta - theta_r)^2 dx, no load. Plate energy = U - hinge energy."""
    n, An = glauert_coefficients(a, k, D, nu, theta_r=theta_r, N=N)
    tau, wq = np.polynomial.legendre.leggauss(nquad)
    tau = 0.5 * np.pi * (tau + 1.0); wq = 0.5 * np.pi * wq
    th = np.sin(np.outer(tau, n)) @ An
    return 0.5 * k * np.sum(wq * a * np.sin(tau) * (th - theta_r) ** 2)


def fold_energy_release_rate(a, k, D, nu, theta_r, N=800, rel_step=1e-4):
    """G_f per tip = -1/2 dU/da (paper, Section 5.2), by central difference in a.
    Also returns the second route K_M^2/(2 Lam) - 1/2 k theta_r^2 and its two parts."""
    Lam = line_stiffness(D, nu)
    da = rel_step * a
    Gf = -0.5 * (stored_energy_rest_angle(a + da, k, D, nu, theta_r, N)
                 - stored_energy_rest_angle(a - da, k, D, nu, theta_r, N)) / (2.0 * da)
    n, An = glauert_coefficients(a, k, D, nu, theta_r=theta_r, N=N)
    KM = moment_intensity(a, n, An, D, nu)
    plate_part = KM ** 2 / (2.0 * Lam)
    hinge_part = 0.5 * k * theta_r ** 2
    return {"Gf": Gf, "Gf_split": plate_part - hinge_part,
            "plate_part": plate_part, "hinge_part": hinge_part, "K_M": KM}


def deflection(x, y, a, n, An, nu):
    """w(x, y) from the line integral w = int theta(t) K(x - t, y) dt (paper, Section 3 and Appendix B),
    K = (1/8pi)[2(1+nu) ln rho + (1+nu) + 2(y^2 + nu s^2)/rho^2], s = x - t.
    Adaptive quadrature with a break point at t = x (the kernel varies sharply there when y is small).
    Defined up to a rigid-body motion. Infinite plate: w grows like ln r far away,
    so compare shapes after removing a best-fit plane, not absolute values."""
    from scipy.integrate import quad
    def one(xx, yy):
        def f(t):
            s = xx - t; rho2 = s * s + yy * yy
            K = ((1 + nu) * np.log(rho2) + (1 + nu) + 2 * (yy * yy + nu * s * s) / rho2) / (8 * np.pi)
            return theta(t, a, n, An)[0] * K
        pts = [xx] if -a < xx < a else None
        return quad(f, -a, a, points=pts, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    X, Y = np.broadcast_arrays(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)))
    return np.vectorize(one)(X, Y)


if __name__ == "__main__":
    # Self-checks (D = 1, nu = 0.3, a = 1)
    D, nu, a = 1.0, 0.3, 1.0
    Lam = line_stiffness(D, nu)
    # free hinge under M0: theta(0) = -2 M0 a / Lam, K_M = M0 sqrt(pi a), M_yy ahead = M0 x / sqrt(x^2 - a^2)
    n, An = glauert_coefficients(a, 1e-12, D, nu, M0=1.0, N=50)
    print("free hinge theta(0):", theta(0.0, a, n, An)[0], "expect", -2 * a / Lam)
    print("free hinge K_M:", moment_intensity(a, n, An, D, nu), "expect", np.sqrt(np.pi * a))
    xs = np.array([1.2, 2.0, -1.5])
    print("free hinge M ahead:", moment_on_line(xs, a, n, An, D, nu, M0=1.0), "expect", np.abs(xs) / np.sqrt(xs ** 2 - 1))
    # rest angle, Table 1 of the paper: G_f / (k theta_r^2 / 2) at lam = 0.1, 1, 10
    for lam, ref in [(0.1, -0.7508), (1.0, -0.1868), (10.0, -0.0169)]:
        k = lam * Lam / a
        r = fold_energy_release_rate(a, k, D, nu, theta_r=1.0)
        print(f"lam={lam}: Gf/(k/2)={r['Gf']/(0.5*k):.4f} (table {ref}), split={r['Gf_split']/(0.5*k):.4f}")
    # slope jump of w equals theta
    n, An = glauert_coefficients(a, Lam, D, nu, theta_r=1.0, N=200)
    e, d = 1e-4, 1e-4
    wy_up = (deflection(0.3, e + d, a, n, An, nu) - deflection(0.3, e, a, n, An, nu)) / d
    wy_dn = (deflection(0.3, -e, a, n, An, nu) - deflection(0.3, -e - d, a, n, An, nu)) / d
    print("slope jump at x=0.3:", (wy_up - wy_dn)[0], "theta:", theta(0.3, a, n, An)[0])
    # energy split, lam = 1
    k = Lam / a
    U = stored_energy_rest_angle(a, k, D, nu, 1.0); Eh = hinge_energy(a, k, D, nu, 1.0)
    print(f"lam=1: U={U:.5f}, hinge={Eh:.5f}, plate={U-Eh:.5f} (U/Lam = {U/Lam:.4f}, table g=0.4116)")


# Shorter name used by the scripts.
stored_energy = stored_energy_rest_angle
