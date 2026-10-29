"""Exact optimal coefficients and norms: implicit m=7 (k=4 classical, 5, 6) and m=8 (k=5 classical, 7)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as F
from math import lcm
import numpy as np
from _lib import optimal, Cvec
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_tests.py")).read()
exec(src[src.index("def lmm("):src.index("res = {")])

def exact_degree(c, k):
    C = Cvec(k); d = 1
    while True:
        a = d + 1
        if sum(C[b] * b ** a for b in range(k + 1)) != a * (a - 1) * sum(c[b] * F(b) ** (a - 2) for b in range(len(c))):
            return d
        d += 1

def err_const(c, k):
    p = exact_degree(c, k) - 1; a = p + 2
    C = Cvec(k)
    # local error = Ccoef * h^{p+2} u^{(p+2)}; Ccoef = [sum C b^a - a(a-1) sum c b^{a-2}]/a!
    from math import factorial
    v = sum(C[b] * F(b) ** a for b in range(k + 1)) - a * (a - 1) * sum(c[b] * F(b) ** (a - 2) for b in range(len(c)))
    return p, v / factorial(a)

def stab(c, Hmax=12, n=4800):
    k = len(c) - 1; last = 0
    for H2 in np.linspace(Hmax / n, Hmax, n):
        p = np.zeros(k + 1); p[k] += 1; p[k - 1] -= 2; p[k - 2] += 1; p = p + H2 * np.array(c)
        if np.max(np.abs(np.roots(p[::-1]))) > 1 + 1e-9: break
        last = H2
    return last

res = {}
for m, k in [(7, 4), (7, 5), (7, 6), (8, 5), (8, 7)]:
    c, nn = optimal(m, k, False)
    D = lcm(*[x.denominator for x in c])
    p, ec = err_const(c, k)
    cf = [float(x) for x in c]
    res[(m, k)] = (c, nn, cf)
    print(f"\n=== m={m}, k={k} ({'classical Cowell' if k == m - 3 else 'optimal'}) ===")
    print("  C^||[0..k] =", [str(x) for x in c])
    print("  common denominator D =", D, " numerators:", [int(x * D) for x in c], " sum =", sum(c))
    print(f"  ||l||^2 = {nn} h^{2*m-1} = {float(nn):.6e} h^{2*m-1}")
    print(f"  order {p}, local error constant (coeff of h^{p+2} u^({p+2})) = {ec} = {float(ec):.4e}")
    print(f"  stability interval H^2 < {stab(cf):.3f}")

print("\nKEPLER e=0.5, max position error (10 / 100 periods)")
T = 2 * np.pi
for spp in (100, 200):
    h = T / spp
    for (m, k), (c, nn, cf) in res.items():
        out = []
        for P in (10, 100):
            t, Y, _ = lmm(np.array(cf), kep_f, kep_q, h, P * spp)
            out.append(np.max(np.linalg.norm(Y - np.array([kep_q(tt) for tt in t]), axis=1)))
        print(f"  spp={spp} m={m} k={k}: {out[0]:.2e} / {out[1]:.2e}")
print("\nOSCILLATOR y''=-100y, h=1/40, T=100")
w = 10.0; h = 1 / 40; n = 4000
for (m, k), (c, nn, cf) in res.items():
    t, Y, _ = lmm(np.array(cf), lambda y: -w * w * y, lambda t: np.array([np.cos(w * t)]), h, n)
    print(f"  m={m} k={k}: err={np.max(np.abs(Y[:,0]-np.cos(w*t))):.2e}")
