"""Table 2 (Section 6) and Appendix D (Table D.6): numerical checks of the analytical results.

Each check compares a result with an independent one. D = 1, nu = 0.3, a = 1 unless stated.
Run time: about 10-30 minutes (the domain integrals use adaptive quadrature).
Run one check with:  python checks.py 6
"""
import sys
import warnings
import numpy as np
import sympy as sp
from scipy.integrate import quad
from _common import NU, FiniteSheet
from creasefm import (line_stiffness, glauert_coefficients, theta, fold_energy_release_rate,
                      moment_intensity)
from creasefm.finite_sheet import winf_derivs

warnings.filterwarnings("ignore", message="The occurrence of roundoff error")

D, nu, a = 1.0, NU, 1.0
Lam = line_stiffness(D, nu)


# ---------------------------------------------------------------- 1, 2: uniform crease, closed form
th0 = 0.1


def w_uniform(x, y):
    """Deflection of a uniform crease of angle th0 on |x| < a (paper, Section 3.3)."""
    r1, r2 = np.hypot(x - a, y), np.hypot(x + a, y)
    p1, p2 = np.arctan2(y, x - a), np.arctan2(y, x + a)
    return th0 / (4 * np.pi) * ((1 + nu) * ((x + a) * np.log(r2) - (x - a) * np.log(r1)) + 2 * y * (p1 - p2))


def check1():
    print("1. Slope jump of the uniform-crease field, theta_0 = 0.1")
    for x0 in [0.0, 0.5, 1.5]:
        e, d = 1e-6, 1e-5
        jump = (w_uniform(x0, e + d) - w_uniform(x0, e)) / d - (w_uniform(x0, -e) - w_uniform(x0, -e - d)) / d
        print(f"   x = {x0}: jump = {jump:.6f}  (expect {th0 if abs(x0) < a else 0})")


def check2():
    print("2. Moment on the line, Eq. (Mbc), against finite differences of w")
    for x0 in [0.0, 0.6, 1.4, 2.5]:
        y0, h = 1e-4, 2e-5
        fxx = (w_uniform(x0 + h, y0) - 2 * w_uniform(x0, y0) + w_uniform(x0 - h, y0)) / h ** 2
        fyy = (w_uniform(x0, y0 + 2e-5) - 2 * w_uniform(x0, y0 + 1e-5) + w_uniform(x0, y0)) / 1e-10
        M22 = -D * (fyy + nu * fxx)
        print(f"   x = {x0}: finite differences {M22:.5f}, formula {Lam * th0 * a / (np.pi * (a ** 2 - x0 ** 2)):.5f}")


# ---------------------------------------------------------------- 3, 4: free hinge and Glauert limits
def check3():
    print("3. Free hinge under M0 = 1: residual of the hinge condition M_yy = 0 on the crease")
    N = 40
    sn = np.cos((np.arange(1, N + 1) - 0.5) * np.pi / N)          # Gauss-Chebyshev nodes
    om = lambda t: (2.0 / Lam) * t * (-1.0)                        # omega = -theta' times sqrt(a^2 - t^2)
    xm = np.cos(np.arange(1, N) * np.pi / N)
    res = [1.0 + Lam / 2 * np.mean(om(sn) / (sn - x)) for x in xm]
    print(f"   max |M_yy| on the crease = {np.max(np.abs(res)):.1e}")


def check4():
    print("4. Glauert solver, k -> 0 and k -> infinity")
    n, A = glauert_coefficients(a, 1e-9, D, nu, theta_r=0.0, M0=1.0, N=200)
    print(f"   k -> 0:   theta(0) = {theta(0.0, a, n, A)[0]:.10f}, free hinge {-2 * a / Lam:.10f}")
    n, A = glauert_coefficients(a, 1e6, D, nu, theta_r=th0, M0=0.0, N=400)
    print(f"   k -> inf: theta(0), theta(0.5) = {theta(0.0, a, n, A)[0]:.6f}, {theta(0.5, a, n, A)[0]:.6f}"
          f"  (expect {th0})")


# ---------------------------------------------------------------- 5: two routes to G_f
def check5():
    print("5. G_f / (k theta_r^2 / 2) by -1/2 dU/da and by K_M^2/(2 Lam) - k theta_r^2/2")
    for lam in [0.1, 1, 10]:
        k = lam * Lam / a
        r = fold_energy_release_rate(a, k, D, nu, 1.0, N=800)
        print(f"   lambda = {lam}: {r['Gf'] / (0.5 * k):.5f} / {r['Gf_split'] / (0.5 * k):.5f}")


# ---------------------------------------------------------------- 6: J* on circles
_x, _y, _c, _nu = sp.symbols("x y c nu", real=True)
_X = _x - _c
_H = (1 + _nu) * _X * sp.log(sp.sqrt(_X ** 2 + _y ** 2)) - 2 * _y * sp.atan2(_y, _X)
_HD = {(i, j): sp.lambdify((_x, _y, _c, _nu), sp.diff(_H, _x, i, _y, j) if j else sp.diff(_H, _x, i), "numpy")
       for i in range(4) for j in range(4 - i) if i + j >= 1}
_tq, _wq = np.polynomial.legendre.leggauss(600)
_tq = 0.5 * np.pi * (_tq + 1)
_wq = 0.5 * np.pi * _wq


def _field(px, py, n, A):
    """Derivatives of w at (px, py): w = -(1/4 pi) int omega(c) H_c dc, with omega = -theta'."""
    out = {}
    if abs(py) > 0.08:
        cs = a * np.cos(_tq)
        g = (np.cos(np.outer(_tq, n)) @ (n * A)) * _wq
        for key, f in _HD.items():
            out[key] = -(1 / (4 * np.pi)) * np.sum(g * f(px, py, cs, nu))
        return out
    gfun = lambda t: np.dot(n * A, np.cos(n * t))
    pts = [np.arccos(np.clip(px / a, -1, 1))] if abs(px) < a else None
    for key, f in _HD.items():
        out[key] = -(1 / (4 * np.pi)) * quad(lambda t: gfun(t) * f(px, py, a * np.cos(t), nu), 0, np.pi,
                                              points=pts, limit=800, epsabs=1e-12, epsrel=1e-10)[0]
    return out


def J_circle(rho, n, A, M0=0.0, nphi=2001):
    """Plain J on a circle of radius rho around the tip (a, 0), from the lower face to the upper face.
    M0: remote moment M_yy, added as a uniform field."""
    eta = 2e-4
    phis = np.linspace(-np.pi + eta, np.pi - eta, nphi)
    c0 = M0 / (D * (1 - nu ** 2))
    vals = []
    for ph in phis:
        px = a + rho * np.cos(ph)
        d = _field(px, rho * np.sin(ph), n, A)
        k11, k22, k12 = d[(2, 0)] + nu * c0, d[(0, 2)] - c0, d[(1, 1)]
        w1 = d[(1, 0)] + nu * c0 * px
        M11, M22, M12 = -D * (k11 + nu * k22), -D * (k22 + nu * k11), -D * (1 - nu) * k12
        dM1 = -D * ((d[(3, 0)] + nu * d[(1, 2)]) + (1 - nu) * d[(1, 2)])     # M11,1 + M12,2
        dM2 = -D * ((1 - nu) * d[(2, 1)] + (d[(0, 3)] + nu * d[(2, 1)]))     # M12,1 + M22,2
        W = 0.5 * D * ((k11 + k22) ** 2 - 2 * (1 - nu) * (k11 * k22 - k12 ** 2))
        n1, n2 = np.cos(ph), np.sin(ph)
        vals.append(W * n1 + (M11 * n1 + M12 * n2) * k11 + (M12 * n1 + M22 * n2) * k12
                    - (dM1 * n1 + dM2 * n2) * w1)
    return np.trapezoid(vals, phis) * rho


def check6():
    print("6. Table 2 (contour form) and Appendix D: J, -psi, [M_xy] w_,x and J* = J - psi + [M_xy] w_,x"
          " on circles of radius rho around the tip")
    for case in ["rest", "free"]:
        k, thr, M0 = (Lam / a, 1.0, 0.0) if case == "rest" else (0.0, 0.0, 1.0)
        n, A = glauert_coefficients(a, max(k, 1e-12), D, nu, theta_r=thr, M0=M0, N=200 if case == "rest" else 40)
        exact = fold_energy_release_rate(a, k, D, nu, 1.0)["Gf"] if case == "rest" else np.pi * a / (2 * Lam)

        def dtheta(xc):
            tau = np.arccos(xc / a)
            return np.dot(n * A, np.cos(n * tau)) / (-a * np.sin(tau))

        def wx_line(xc):
            gfun = lambda t: np.dot(n * A, np.cos(n * t))
            f = _HD[(1, 0)]
            return -(1 / (4 * np.pi)) * quad(lambda t: gfun(t) * f(xc, 1e-7, a * np.cos(t), nu), 0, np.pi,
                                             points=[np.arccos(np.clip(xc / a, -1, 1))], limit=800)[0] \
                + nu * M0 / (D * (1 - nu ** 2)) * xc
        for rho in ([0.1, 0.2, 0.4, 0.7] if case == "rest" else [0.2, 0.5]):
            xc = a - rho
            psi = 0.5 * k * (theta(xc, a, n, A)[0] - thr) ** 2
            jump = -D * (1 - nu) * dtheta(xc) * wx_line(xc)        # [M_xy] w_,x at the crossing point
            J = J_circle(rho, n, A, M0=M0)
            print(f"   {case}, rho = {rho}: J = {J:.5f}, -psi = {-psi:.5f}, [M_xy]w_x = {jump:.5f},"
                  f" J* = {J - psi + jump:.5f}  (exact {exact:.5f})", flush=True)


# ---------------------------------------------------------------- 7: domain form of J*
def check7():
    print("7. Table 2 (domain form) and Appendix D: Eq. (Jdomain) on three annuli: rest angle (lambda = 1) and free hinge (M0 = 1)")
    s, y = sp.symbols("s y")
    K = ((1 + nu) * sp.log(s ** 2 + y ** 2) + (1 + nu) + 2 * (y ** 2 + nu * s ** 2) / (s ** 2 + y ** 2)) / (8 * sp.pi)
    der = {key: sp.lambdify((s, y), sp.diff(K, s, key[0], y, key[1]), "math")
           for key in [(1, 0), (2, 0), (1, 1), (0, 2)]}
    for case in ["rest", "free"]:
        k, thr, M0 = (Lam / a, 1.0, 0.0) if case == "rest" else (0.0, 0.0, 1.0)
        target = fold_energy_release_rate(a, k, D, nu, 1.0)["Gf"] if case == "rest" else np.pi * a / (2 * Lam)
        n, A = glauert_coefficients(a, max(k, 1e-14), D, nu, theta_r=thr, M0=M0, N=400)
        th = lambda t: theta(np.atleast_1d(t), a, n, A)[0]

        def wd(X, Y, key):
            f = der[key]
            return quad(lambda t: th(t) * f(X - t, Y), -a, a, points=[X] if -a < X < a else None,
                        limit=500, epsabs=1e-12, epsrel=1e-10)[0]

        for r1, r2 in [(0.1, 0.3), (0.2, 0.5), (0.3, 0.7)]:
            gr, gw = np.polynomial.legendre.leggauss(24)
            gp, gpw = np.polynomial.legendre.leggauss(64)
            R = r1 + (r2 - r1) * (gr + 1) / 2
            WR = gw * (r2 - r1) / 2
            tot = 0.0
            for rho, wr in zip(R, WR):
                sv = (rho - r1) / (r2 - r1)
                dq = (-6 * sv + 6 * sv ** 2) / (r2 - r1)          # q = 1 - 3 s^2 + 2 s^3
                d2q = (-6 + 12 * sv) / (r2 - r1) ** 2
                for ph, wp in zip(np.pi * gp, gpw * np.pi):
                    X, Y = a + rho * np.cos(ph), rho * np.sin(ph)
                    if abs(Y) < 1e-12:
                        Y = 1e-12 * np.sign(ph or 1)
                    wx, wxx, wxy, wyy = (wd(X, Y, kk) for kk in [(1, 0), (2, 0), (1, 1), (0, 2)])
                    c0 = M0 / (D * (1 - nu ** 2))                  # uniform field of the remote moment
                    wxx += nu * c0; wyy += -c0; wx += nu * c0 * X
                    M = np.array([[-D * (wxx + nu * wyy), -D * (1 - nu) * wxy],
                                  [-D * (1 - nu) * wxy, -D * (wyy + nu * wxx)]])
                    W = 0.5 * D * ((wxx + wyy) ** 2 - 2 * (1 - nu) * (wxx * wyy - wxy ** 2))
                    e = np.array([np.cos(ph), np.sin(ph)])
                    gq = dq * e
                    Hq = d2q * np.outer(e, e) + (dq / rho) * (np.eye(2) - np.outer(e, e))
                    tot += (W * gq[0] + 2 * np.array([wxx, wxy]) @ M @ gq + wx * np.sum(M * Hq)) * rho * wr * wp
            line = quad(lambda r: 0.5 * k * (th(a - r) - thr) ** 2
                        * ((-6 * ((r - r1) / (r2 - r1)) + 6 * ((r - r1) / (r2 - r1)) ** 2) / (r2 - r1)),
                        r1, r2, limit=200)[0]
            print(f"   {case}, annulus {r1}-{r2}: area {-tot:.5f}, line {line:.5f},"
                  f" J* = {-tot + line:.5f}  (exact {target:.5f})", flush=True)


# ---------------------------------------------------------------- 8: integration by parts, Eq. (fs_B)
def check8():
    print("8. Integration by parts with Eq. (fs_B): a(u,v) - int D Lap^2 u v = B(u,v), random polynomials")
    x, y = sp.symbols("x y")
    rng = np.random.default_rng(0)
    rpoly = lambda deg: sum(rng.normal() * x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i))
    g, gw = np.polynomial.legendre.leggauss(40)

    def area(f):
        F = sp.lambdify((x, y), f, "numpy")
        Xg, Yg = np.meshgrid(g, g)
        return np.sum(np.outer(gw, gw) * F(Xg, Yg) * np.ones_like(Xg))

    def a_form(u, v):
        return area(D * ((1 - nu) * (sp.diff(u, x, 2) * sp.diff(v, x, 2) + 2 * sp.diff(u, x, y) * sp.diff(v, x, y)
                                     + sp.diff(u, y, 2) * sp.diff(v, y, 2))
                         + nu * (sp.diff(u, x, 2) + sp.diff(u, y, 2)) * (sp.diff(v, x, 2) + sp.diff(v, y, 2))))

    def bterms(u, v):
        lap = sp.diff(u, x, 2) + sp.diff(u, y, 2)
        T1 = T2 = 0.0
        for (nx, ny), (sx, sy), fix, val in [((1, 0), (0, 1), "x", 1.0), ((-1, 0), (0, -1), "x", -1.0),
                                             ((0, 1), (-1, 0), "y", 1.0), ((0, -1), (1, 0), "y", -1.0)]:
            dn = lambda f: nx * sp.diff(f, x) + ny * sp.diff(f, y)
            ds = lambda f: sx * sp.diff(f, x) + sy * sp.diff(f, y)
            Mn = -D * (dn(dn(u)) + nu * ds(ds(u)))
            Vn = -D * dn(lap) + ds(-D * (1 - nu) * dn(ds(u)))
            for e, which in [(Mn * dn(v), 1), (Vn * v, 2)]:
                F = sp.lambdify(y, e.subs(x, val), "numpy") if fix == "x" else sp.lambdify(x, e.subs(y, val), "numpy")
                s_ = np.sum(gw * F(g) * np.ones_like(g))
                if which == 1:
                    T1 += s_
                else:
                    T2 += s_
        Mxy = -D * (1 - nu) * sp.diff(u, x, y)
        T3 = sum(float((2 * Mxy * v).subs({x: cx, y: cy})) * cx * cy for cx, cy in [(1, 1), (-1, 1), (-1, -1), (1, -1)])
        return T1, T2, T3

    rows, rhs = [], []
    for _ in range(8):
        u, v = rpoly(5), rpoly(4)
        rhs.append(a_form(u, v) - area(sp.expand(D * (sp.diff(u, x, 4) + 2 * sp.diff(u, x, 2, y, 2) + sp.diff(u, y, 4))) * v))
        rows.append(bterms(u, v))
    coef = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    print(f"   fitted signs of the edge-moment, edge-shear and corner terms: {np.round(coef, 10)}  (expect -1, 1, -1)")


# ---------------------------------------------------------------- 9: finite-sheet energy
def check9():
    print("9. Finite sheet: U by Eq. (Pirest) against the direct energy sum (plate + hinge)")
    Dc, th = 1.0989010989, np.deg2rad(0.5)
    fs = FiniteSheet(D=Dc, nu=nu, P=20)

    def Bself(aa, A):
        W, s, wq, tot, Ab = 1.0, fs.ge, fs.gwe, 0.0, A[:fs.nb]
        for (nx, ny) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            X, Y = (np.full_like(s, nx * W), s) if nx else (s, np.full_like(s, ny * W))
            d = winf_derivs(X, Y, aa, fs.nb, nu)
            if nx:
                Mn = -Dc * (d[(2, 0)] + nu * d[(0, 2)])
                Vn = -Dc * nx * (d[(3, 0)] + d[(1, 2)]) - Dc * (1 - nu) * nx * d[(1, 2)]
                vn = nx * d[(1, 0)]
            else:
                Mn = -Dc * (d[(0, 2)] + nu * d[(2, 0)])
                Vn = -Dc * ny * (d[(2, 1)] + d[(0, 3)]) - Dc * (1 - nu) * ny * d[(2, 1)]
                vn = ny * d[(0, 1)]
            tot += np.sum(wq * (-(Mn @ Ab) * (vn @ Ab) + (Vn @ Ab) * (d[(0, 0)] @ Ab)))
        Xc, Yc = np.array([W, -W, -W, W]), np.array([W, W, -W, -W])
        d = winf_derivs(Xc, Yc, aa, fs.nb, nu)
        return tot - np.sum(Xc * Yc * 2 * (-Dc * (1 - nu) * d[(1, 1)] @ Ab) * (d[(0, 0)] @ Ab))

    for xi in [1, 100]:
        k = xi * Dc
        for aa in [0.25, 0.4]:
            n, A = fs.solve(aa, k, th, N=600)
            c = fs.last["Cw"] @ A[:fs.nb]
            tg, tw = np.polynomial.legendre.leggauss(800)
            tau = 0.5 * np.pi * (tg + 1); tw = 0.5 * np.pi * tw
            xq = aa * np.cos(tau); jac = aa * np.sin(tau)
            thq = np.sin(np.outer(tau, n)) @ A
            dv = fs._basis_derivs(xq, 0 * xq)
            Mc = -Dc * (dv["yy"] + nu * dv["xx"]) @ c
            Minf = -k * (thq - th) - Mc
            Uh = 0.5 * k * np.sum(tw * jac * (thq - th) ** 2)
            Up = 0.5 * np.sum(tw * jac * Minf * thq) + 0.5 * Bself(aa, A) - 0.5 * c @ fs.Kmat @ c
            short = 0.5 * k * th * np.sum(tw * jac * (th - thq))
            print(f"   xi = {xi}, a = {aa}: direct {Uh + Up:.8e}, Eq. (Pirest) {short:.8e}")


CHECKS = [check1, check2, check3, check4, check5, check6, check7, check8, check9]
if __name__ == "__main__":
    pick = [int(s) for s in sys.argv[1:]] or range(1, 10)
    for i in pick:
        CHECKS[i - 1]()
