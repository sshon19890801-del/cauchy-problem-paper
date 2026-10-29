"""Same stencil: optimal (order m-2) vs classical maximal-order scheme on the same k."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction as F
from _lib import optimal, maximal_order
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_tests.py")).read()
exec(src[src.index("def lmm("):src.index("res = {")])
cases = [("opt m=10 impl k=9 (o8)", optimal(10, 9, False)[0]),
         ("Cowell k=9 (o10)", maximal_order(9, False)),
         ("opt m=8 expl k=8 (o6)", optimal(8, 8, True)[0] + [0]),
         ("opt m=10 expl k=10 (o8)", optimal(10, 10, True)[0] + [0]),
         ("Stormer k=8 (o8)", maximal_order(8, True) + [0]),
         ("Stormer k=10 (o10)", maximal_order(10, True) + [0])]
T = 2 * np.pi
for spp in (200, 400):
    h = T / spp
    for name, c in cases:
        cf = np.array([float(x) for x in c]); out = []
        for P in (10, 100):
            t, Y, _ = lmm(cf, kep_f, kep_q, h, P * spp)
            ok = np.all(np.isfinite(Y)) and np.max(np.abs(Y)) < 1e6
            out.append(np.max(np.linalg.norm(Y - np.array([kep_q(tt) for tt in t]), axis=1)) if ok else np.inf)
        print(f"spp={spp} {name:24s} err10={out[0]:9.2e} err100={out[1]:9.2e}")
