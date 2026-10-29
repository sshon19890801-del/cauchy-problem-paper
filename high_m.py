"""Optimal schemes for m = 8, 9, 10 vs classical Stormer/Cowell of the same order (Kepler, oscillator)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction as F
from _lib import optimal, maximal_order, norm2, Cvec

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_tests.py")).read()
a = src.index("def lmm("); b = src.index("res = {")
exec(src[a:b])  # lmm, rk4, Kepler functions

def stab(c, Hmax=40.0, n=4000):
    """largest H^2 such that all roots |zeta|<=1+1e-9 on (0,H^2]"""
    k = len(c) - 1
    last = 0.0
    for H2 in np.linspace(Hmax / n, Hmax, n):
        p = np.zeros(k + 1); p[k] += 1; p[k - 1] -= 2; p[k - 2] += 1
        p = p + H2 * np.array(c)
        if np.max(np.abs(np.roots(p[::-1]))) > 1 + 1e-9:
            break
        last = H2
    return last

rows = []
for m in (8, 9, 10):
    order = m - 2
    for explicit in (True, False):
        kmin = m - 2 if explicit else m - 3
        for k in (kmin, kmin + 2, kmin + 4):
            c, nn = optimal(m, k, explicit)
            if explicit: c = c + [F(0)]
            cf = np.array([float(x) for x in c])
            label = ("classical " if k == kmin else "optimal   ") + ("expl" if explicit else "impl")
            rows.append((m, order, label, k, float(nn), cf))

print("m ord scheme          k   norm2/h^(2m-1)   max|C|   stab H^2")
for m, o, lab, k, nn, cf in rows:
    print(f"{m:2d} {o:2d}  {lab:15s} {k:2d}  {nn:12.4e}  {np.max(np.abs(cf)):8.2f}  {stab(cf):6.2f}")

print("\nKEPLER e=0.5, max position error")
T = 2 * np.pi
for spp in (100, 200):
    h = T / spp
    print(f"--- {spp} steps/period")
    for m, o, lab, k, nn, cf in rows:
        out = []
        for P in (10, 100):
            t, Y, _ = lmm(cf, kep_f, kep_q, h, P * spp)
            if not np.all(np.isfinite(Y)) or np.max(np.abs(Y)) > 1e6:
                out.append(np.inf); continue
            ex = np.array([kep_q(tt) for tt in t])
            out.append(np.max(np.linalg.norm(Y - ex, axis=1)))
        print(f"  m={m:2d} ord={o} {lab:15s} k={k:2d}  err10={out[0]:9.2e}  err100={out[1]:9.2e}")

print("\nOSCILLATOR y''=-100y, h=1/40, T=100")
w = 10.0
h = 1 / 40; n = int(round(100 / h))
for m, o, lab, k, nn, cf in rows:
    t, Y, _ = lmm(cf, lambda y: -w * w * y, lambda t: np.array([np.cos(w * t)]), h, n)
    err = np.max(np.abs(Y[:, 0] - np.cos(w * t))) if np.all(np.isfinite(Y)) else np.inf
    print(f"  m={m:2d} ord={o} {lab:15s} k={k:2d}  err={err:9.2e}")
