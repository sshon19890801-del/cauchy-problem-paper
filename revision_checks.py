"""Additional checks added in the revision (Python 3, numpy, matplotlib; exact rationals via fractions).

  1. simplified closed form of Theorem 4 against the direct minimisation, k = 3..10
  2. structure of Theorems 1 and 2 for m = 5, 6 (non-trivial cases with free parameters)
  3. optimal symmetric two-step scheme in L_2^(4)
  4. norms in L_2^(4)* of classical schemes (Verlet, Numerov, 6-step Stormer and Cowell)
  5. work-precision data for problem (84) -> workprec.json, figure2.png
"""
from fractions import Fraction as F
from math import factorial, sqrt, comb
import json
import numpy as np


def G(m, d, x):
    p = 2 * m - 1 - d
    return F(abs(x) ** p, 2 * factorial(p))


def gauss(A, b):
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for i in range(n):
        p = next(r for r in range(i, n) if M[r][i] != 0)
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i and M[r][i] != 0:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


def norm2(m, C, c):
    s = (-1) ** m
    K, n = len(C), len(c)
    a = sum(C[i] * C[j] * G(m, 0, i - j) for i in range(K) for j in range(K))
    b = sum(C[i] * c[j] * G(m, 2, i - j) for i in range(K) for j in range(n))
    d = sum(c[i] * c[j] * G(m, 4, i - j) for i in range(n) for j in range(n))
    return s * (a - 2 * b + d)


def Cvec(k):
    C = [0] * (k + 1)
    C[k - 2], C[k - 1], C[k] = 1, -2, 1
    return C


def optimal(m, k, explicit):
    C = Cvec(k)
    n = k if explicit else k + 1
    s = (-1) ** m
    bv = [sum(C[i] * G(m, 2, i - g) for i in range(k + 1)) for g in range(n)]
    H = [[G(m, 4, i - j) for j in range(n)] for i in range(n)]
    rows = [[F(b) ** (a - 2) for b in range(n)] for a in range(2, m)]
    rhs = [F(sum(C[b] * b ** a for b in range(k + 1)), a * (a - 1)) for a in range(2, m)]
    nc = len(rows)
    A = [[2 * s * H[i][j] for j in range(n)] + [-rows[l][i] for l in range(nc)] for i in range(n)]
    A += [rows[l] + [F(0)] * nc for l in range(nc)]
    x = gauss(A, [2 * s * bb for bb in bv] + rhs)
    c = x[:n]
    return c, norm2(m, C, c)


def maximal_order(k, explicit):
    """classical Stormer (explicit) / Cowell (implicit) k-step scheme: exactness of maximal degree"""
    C = Cvec(k)
    n = k if explicit else k + 1
    rows = [[F(b) ** (a - 2) for b in range(n)] for a in range(2, n + 2)]
    rhs = [F(sum(C[b] * b ** a for b in range(k + 1)), a * (a - 1)) for a in range(2, n + 2)]
    return gauss(rows, rhs)


out = {}

# 1. Theorem 4, simplified form
q = sqrt(3) - 2
print("1. Theorem 4 (simplified) vs direct minimisation")
t4 = []
for k in range(3, 11):
    simp = sqrt(3) / 150 - 131 / 12600 + q ** (2 * k) / (25 * sqrt(3) * (1 - q ** (2 * k)))
    _, nd = optimal(4, k, False)
    t4.append((k, str(nd), float(nd), simp))
    print(f"  k={k:2d}  direct={nd}  ={float(nd):.15e}  formula={simp:.15e}")
out["thm4"] = t4
print("  limit k->inf:", sqrt(3) / 150 - 131 / 12600)


# 2. Structure of Theorems 1-2 for m = 5, 6
def eulerian_poly(n):
    # E_n(x) = sum_j A(n+1, j) x^j  (E_2 = x^2+4x+1)
    N = n + 1
    return [sum((-1) ** i * comb(N + 1, i) * (j + 1 - i) ** N for i in range(j + 2)) for j in range(n + 1)]


print("2. Structure of Theorems 1 and 2")
struct = []
for m in (5, 6):
    coeffs = eulerian_poly(2 * m - 6)
    roots = np.roots(coeffs[::-1])
    qs = np.sort(np.real(roots[np.abs(roots) < 1]))
    for explicit in (False, True):
        k = 12
        c, nd = optimal(m, k, explicit)
        cf = np.array([float(x) for x in c])
        # fit m_i, n_i from interior coefficients beta = 1..k-3
        bet = np.arange(1, k - 2)
        A = np.column_stack([qs[i] ** bet for i in range(m - 3)] + [qs[i] ** (k - bet - 1) for i in range(m - 3)])
        sol, *_ = np.linalg.lstsq(A, cf[1:k - 2], rcond=None)
        mi, ni = sol[:m - 3], sol[m - 3:]
        res_int = np.max(np.abs(A @ sol - cf[1:k - 2]))
        expo = lambda b: float(np.sum(mi * qs ** b + ni * qs ** (k - b - 1)))
        d = (2 * m - 4) * (2 * m - 3)
        rec = {"m": m, "explicit": explicit, "k": k, "roots": qs.tolist(), "interior_residual": res_int}
        rec["C[k-2] constant error"] = cf[k - 2] - expo(k - 2) - 1 / d
        if not explicit:
            rec["C[k-1] constant error"] = cf[k - 1] - expo(k - 1) - (3 * 2 ** (2 * m - 5) - 2) / d
            Amk = ((k - 1) * (d - 3 * 2 ** (2 * m - 5)) + k) / (d * k)
            Bmk = (1 - 3 * 2 ** (2 * m - 5) / d) / k
            Ck = Amk + np.sum((mi - ni) * (qs ** k - qs) / (k * (1 - qs) ** 2) + (k - 1) * (mi * qs ** k - ni) / (k * (1 - qs)))
            C0 = Bmk + np.sum((ni - mi) * (qs ** k - qs) / (k * (1 - qs) ** 2) + (mi * qs ** k - ni) / (k * (1 - qs)) + (mi * qs - ni * qs ** (k - 1)) / (qs - 1))
            rec["C[k] formula - direct"] = float(Ck - cf[k])
            rec["C[0] formula - direct"] = float(C0 - cf[0])
        else:
            Amk = 1 - (k - 2) / (d * (k - 1))
            Bmk = -1 / (d * (k - 1))
            Ck1 = Amk + np.sum((mi - ni) * (qs ** k - qs) / ((k - 1) * (1 - qs) ** 2) + (mi * qs ** (k - 1) - ni * qs) / (1 - qs))
            C0 = Bmk - np.sum((mi - ni) * (qs ** k - qs) / ((k - 1) * (1 - qs) ** 2) + (mi * qs - ni * qs ** (k - 1)) / (1 - qs))
            rec["C[k-1] formula - direct"] = float(Ck1 - cf[k - 1])
            rec["C[0] formula - direct"] = float(C0 - cf[0])
        struct.append(rec)
        print("  ", {kk: (vv if not isinstance(vv, float) else f"{vv:.3e}") for kk, vv in rec.items()})
out["structure"] = struct

# 3. symmetric two-step scheme in L_2^(4): c = (a, 1-2a, a)
print("3. Symmetric two-step family in L_2^(4)")
n0 = norm2(4, [1, -2, 1], [F(0), F(1), F(0)])
n1 = norm2(4, [1, -2, 1], [F(1), F(-1), F(1)])  # a = 1
n2 = norm2(4, [1, -2, 1], [F(2), F(-3), F(2)])  # a = 2
# quadratic Q(a) = n0 + B a + A a^2
A_ = (n2 - 2 * n1 + n0) / 2
B_ = n1 - n0 - A_
a_opt = -B_ / (2 * A_)
c_sym = [a_opt, 1 - 2 * a_opt, a_opt]
n_sym = norm2(4, [1, -2, 1], c_sym)
print("  a_opt =", a_opt, " norm2 =", n_sym, float(n_sym))
out["symmetric"] = {"a_opt": str(a_opt), "norm2": str(n_sym), "float": float(n_sym)}

# 4. norms in L_2^(4)* of classical schemes
print("4. Norms in L_2^(4)*")
st6 = maximal_order(6, True)
cw6 = maximal_order(6, False)
tab = {
    "Stormer-Verlet": norm2(4, [1, -2, 1], [F(0), F(1), F(0)]),
    "Numerov": norm2(4, [1, -2, 1], [F(1, 12), F(10, 12), F(1, 12)]),
    "symmetric optimal": n_sym,
    "opt expl k=6": optimal(4, 6, True)[1],
    "opt impl k=3": optimal(4, 3, False)[1],
    "opt impl k=6": optimal(4, 6, False)[1],
    "Stormer k=6 (order 6)": norm2(4, Cvec(6), st6),
    "Cowell k=6 (order 7)": norm2(4, Cvec(6), cw6),
}
for kk, vv in tab.items():
    print(f"  {kk:24s} {str(vv):>30s}  {float(vv):.6e}")
print("  Stormer k=6 coeffs:", [str(x) for x in st6])
print("  Cowell  k=6 coeffs:", [str(x) for x in cw6])
out["norms_L24"] = {kk: [str(vv), float(vv)] for kk, vv in tab.items()}

# 5. work-precision for problem (84): y'' = x + y, y(0)=2, y'(0)=-1, exact start values
yex = lambda x: np.exp(x) + np.exp(-x) - x


def lmm(cpar, implicit, h):
    cpar = np.array([float(v) for v in cpar])
    k = len(cpar) - 1 if implicit else len(cpar)
    N = int(round(1 / h))
    x = np.arange(N + 1) * h
    y = np.empty(N + 1)
    y[:k] = yex(x[:k])
    for n in range(0, N - k + 1):
        rhs = 2 * y[n + k - 1] - y[n + k - 2]
        for b in range(len(cpar)):
            if implicit and b == k:
                continue
            rhs += h * h * cpar[b] * (x[n + b] + y[n + b])
        if implicit:
            ck = cpar[k]
            y[n + k] = (rhs + h * h * ck * x[n + k]) / (1 - h * h * ck)
        else:
            y[n + k] = rhs
    return np.max(np.abs(y - yex(x))), N


def rk4(h):
    N = int(round(1 / h))
    f = lambda x, z: np.array([z[1], x + z[0]])
    z = np.array([2.0, -1.0]); x = 0.0; err = 0.0
    for _ in range(N):
        k1 = f(x, z); k2 = f(x + h / 2, z + h / 2 * k1); k3 = f(x + h / 2, z + h / 2 * k2); k4 = f(x + h, z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); x += h
        err = max(err, abs(z[0] - yex(x)))
    return err, 4 * N


schemes = {
    "Stormer-Verlet": ([0, 1, 0], True, [1 / 50, 1 / 100, 1 / 200, 1 / 400, 1 / 800]),
    "Opt. impl. m=4, k=6": (optimal(4, 6, False)[0], True, [1 / 50, 1 / 100, 1 / 200, 1 / 400, 1 / 800]),
    "Numerov": ([F(1, 12), F(10, 12), F(1, 12)], True, [1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160]),
    "Opt. expl. m=6, k=4 (Stormer)": (optimal(6, 4, True)[0], False, [1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160, 1 / 320]),
    "Stormer k=6": (st6, False, [1 / 10, 1 / 20, 1 / 40, 1 / 80]),
    "Cowell k=6": (cw6, True, [1 / 10, 1 / 20, 1 / 40]),
}
wp = {}
for name, (cp, imp, hs) in schemes.items():
    wp[name] = [lmm(cp, imp, h) for h in hs]
wp["RK4"] = [rk4(h) for h in [1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160]]
print("5. work-precision (error, f-evaluations)")
for kk, vv in wp.items():
    print("  ", kk, [(f"{e:.2e}", n) for e, n in vv])
out["workprec"] = {kk: [[float(e), int(n)] for e, n in vv] for kk, vv in wp.items()}

json.dump(out, open("code/revision_checks.json", "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.8))
mk = iter("os^vDxP*")
for kk, vv in wp.items():
    e, n = zip(*vv)
    ax.loglog(n, e, marker=next(mk), label=kk)
ax.set_xlabel("number of evaluations of $f$")
ax.set_ylabel("maximum absolute error")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("figure2.png", dpi=200)
print("saved figure2.png")
