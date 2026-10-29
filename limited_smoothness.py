"""Limited-smoothness test: y'' = -y + g(x), exact u(x) = cos x + |x - x0|^s, x0 = 1/pi (not a grid point).
u is in L_2^(m)(0,1) iff s > m - 1/2.  Optimal schemes for the matching m vs classical schemes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction as F
from _lib import optimal, maximal_order

x0 = 1 / np.pi

def make(s):
    u = lambda x: np.cos(x) + np.abs(x - x0) ** s
    g = lambda x: s * (s - 1) * np.abs(x - x0) ** (s - 2) + np.abs(x - x0) ** s  # u'' + u
    return u, g

def run(c, implicit, s, N):
    u, g = make(s)
    c = np.array([float(v) for v in c]); k = len(c) - 1
    h = 1 / N; x = np.arange(N + 1) * h; y = np.empty(N + 1); y[:k] = u(x[:k])
    fx = lambda j: -y[j] + g(x[j])
    for n in range(k, N + 1):
        rhs = 2 * y[n - 1] - y[n - 2] + h * h * sum(c[b] * fx(n - k + b) for b in range(k))
        if c[k] != 0:
            y[n] = (rhs + h * h * c[k] * g(x[n])) / (1 + h * h * c[k])
        else:
            y[n] = rhs
    return np.max(np.abs(y - u(x)))

def schemes(m):
    S = {}
    for kk in (m - 3, m - 2, m - 1):
        c = optimal(m, kk, False)[0]
        S[f"impl m={m} k={kk}" + (" [Cowell]" if kk == m - 3 else " [opt]")] = c
    for kk in (m - 2, m - 1, m):
        c = optimal(m, kk, True)[0] + [F(0)]
        S[f"expl m={m} k={kk}" + (" [Stormer]" if kk == m - 2 else " [opt]")] = c
    # classical schemes of HIGHER order on the longest stencil used above
    S[f"impl Cowell k={m-1} (order {m})"] = maximal_order(m - 1, False)
    S[f"expl Stormer k={m} (order {m})"] = maximal_order(m, True) + [F(0)]
    S["Numerov (order 4)"] = [F(1, 12), F(10, 12), F(1, 12)]
    return S

Ns = [20, 40, 80, 160, 320, 640]
for s, m in [(3.6, 4), (5.6, 6), (7.6, 8)]:
    print(f"\n===== s = {s}: u in L_2^(m) for m <= {m}  (m={m} -> order {m-2}) =====")
    print(f"{'scheme':32s}" + "".join(f"  N={N:<5d}" for N in Ns) + "   rate(last)")
    for name, c in schemes(m).items():
        E = [run(c, None, s, N) for N in Ns]
        rate = np.log2(E[-2] / E[-1]) if E[-1] > 0 else float('nan')
        print(f"{name:32s}" + "".join(f" {e:8.2e}" for e in E) + f"   {rate:5.2f}")
