"""Rough solution spread over the whole interval:
u(x) = sum_{j=1}^{J} j^{-sigma} sin(2 pi j x + phi_j), sigma = m + 0.6  =>  u in L_2^(m)(0,1), not in L_2^(m+1).
y'' = -y + g(x), g = u'' + u.  Exact starting values; implicit schemes solved exactly (linear)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction as F
from _lib import optimal, maximal_order

rng = np.random.default_rng(1)
J = 3000
phi = rng.uniform(0, 2 * np.pi, J)
jj = np.arange(1, J + 1)

def make(sigma):
    a = jj ** (-sigma)
    def u(x):
        return np.sin(2 * np.pi * np.outer(x, jj) + phi) @ a
    def upp(x):
        return np.sin(2 * np.pi * np.outer(x, jj) + phi) @ (-(2 * np.pi * jj) ** 2 * a)
    return u, upp

def run(c, U, G, N):
    c = np.array([float(v) for v in c]); k = len(c) - 1
    h = 1 / N; y = np.empty(N + 1); y[:k] = U[:k]
    for n in range(k, N + 1):
        rhs = 2 * y[n - 1] - y[n - 2] + h * h * sum(c[b] * (-y[n - k + b] + G[n - k + b]) for b in range(k))
        y[n] = (rhs + h * h * c[k] * G[n]) / (1 + h * h * c[k]) if c[k] != 0 else rhs
    return np.max(np.abs(y - U)) / np.max(np.abs(U))

def schemes(m):
    S = {}
    for kk in (m - 3, m - 2, m - 1):
        if kk >= 2:
            S[f"impl m={m} k={kk}" + (" [Cowell]" if kk == m - 3 else " [opt]")] = optimal(m, kk, False)[0]
    for kk in (m - 2, m - 1, m):
        S[f"expl m={m} k={kk}" + (" [Stormer]" if kk == m - 2 else " [opt]")] = optimal(m, kk, True)[0] + [F(0)]
    S[f"impl Cowell k={m-1} (ord {m})"] = maximal_order(m - 1, False)
    S[f"expl Stormer k={m} (ord {m})"] = maximal_order(m, True) + [F(0)]
    S[f"impl Cowell k={m+1} (ord {m+2})"] = maximal_order(m + 1, False)
    return S

Ns = [100, 200, 400, 800, 1600, 3200]
for m in (4, 6, 8):
    sigma = m + 0.6
    u, upp = make(sigma)
    data = {}
    for N in Ns:
        x = np.arange(N + 1) / N; U = u(x); data[N] = (U, upp(x) + U)
    print(f"\n===== m={m}: sigma={sigma}, u in L_2^({m}) only; relative max error =====")
    print(f"{'scheme':28s}" + "".join(f"  N={N:<6d}" for N in Ns))
    for name, c in schemes(m).items():
        E = [run(c, *data[N], N) for N in Ns]
        print(f"{name:28s}" + "".join(f"  {e:8.2e}" for e in E))
