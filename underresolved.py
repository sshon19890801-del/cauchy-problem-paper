"""Under-resolved forcing: y'' = -y + g, g = narrow Gaussian pulse of width w (only a few grid steps).
Reference solution computed with Cowell k=7 on an 8x finer grid (finer grids are dominated by round-off)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction as F
from _lib import optimal, maximal_order

w = 0.01; x0 = 0.4321
g = lambda x: np.exp(-((x - x0) / w) ** 2) / w

def run(c, N, y0=1.0, y1=None):
    c = np.array([float(v) for v in c]); k = len(c) - 1
    h = 1 / N; x = np.arange(N + 1) * h; G = g(x); y = np.empty(N + 1)
    y[:k] = np.cos(x[:k])  # pulse starts far from x=0, so u = cos x there
    for n in range(k, N + 1):
        rhs = 2 * y[n - 1] - y[n - 2] + h * h * sum(c[b] * (-y[n - k + b] + G[n - k + b]) for b in range(k))
        y[n] = (rhs + h * h * c[k] * G[n]) / (1 + h * h * c[k]) if c[k] != 0 else rhs
    return x, y

ref_c = maximal_order(7, False)
Nref = 8 * 1600
xr, yr = run(ref_c, Nref)
def err(c, N):
    x, y = run(c, N); return np.max(np.abs(y - yr[:: Nref // N]))

S = {}
for m in (4, 6, 8):
    for kk in (m - 3, m - 2, m - 1):
        if kk >= 2:
            S[f"impl m={m} k={kk}" + (" [Cowell]" if kk == m - 3 else " [opt]")] = optimal(m, kk, False)[0]
S["Numerov"] = [F(1, 12), F(10, 12), F(1, 12)]
S["impl Cowell k=7 (ord 8)"] = maximal_order(7, False)
Ns = [50, 100, 200, 400, 800, 1600]
print(f"pulse width w={w}; w/h = " + ", ".join(f"{w*N:.1f}" for N in Ns))
print(f"{'scheme':26s}" + "".join(f"  N={N:<6d}" for N in Ns))
for name, c in S.items():
    print(f"{name:26s}" + "".join(f"  {err(c, N):8.2e}" for N in Ns))
