"""Intervals of periodicity / absolute stability (Lambert-Watson) for the schemes,
applied to y'' = -w^2 y.  Characteristic polynomial  zeta^{k-2}(zeta-1)^2 + H^2 sum c_j zeta^j,  H = w h.
Periodicity: principal roots on the unit circle, remaining roots |zeta| < 1 (or <= 1)."""
import mpmath as mp, json
from fractions import Fraction as Fr
from num import coeffs
mp.mp.dps = 30

def roots(k, C, H2):
    p = [mp.mpf(0)]*(k+1)
    p[k] += 1; p[k-1] += -2; p[k-2] += 1
    for j, c in C.items():
        p[j] += H2*mp.mpf(c.numerator)/c.denominator if isinstance(c, Fr) else H2*c
    # polyroots wants highest degree first
    return mp.polyroots(p[::-1], maxsteps=200, extraprec=60)

def periodic(k, C, H2, tol=1e-12):
    rs = roots(k, C, H2)
    mods = sorted([abs(r) for r in rs], reverse=True)
    # need the two largest moduli == 1 (principal pair), the rest <= 1
    return abs(mods[0]-1) < tol and abs(mods[1]-1) < tol and all(m <= 1+tol for m in mods[2:]) \
        and all(abs(r.imag) > 1e-10 for r in rs if abs(abs(r)-1) < tol)  # principal roots strictly complex

def interval(k, C):
    lo, hi = mp.mpf('1e-6'), mp.mpf(0)
    H2 = mp.mpf('0.001'); last = 0
    while H2 < 100:
        if periodic(k, C, H2): last = H2
        else: break
        H2 *= mp.mpf('1.02')
    if last == 0: return 0
    a, b = last, last*mp.mpf('1.02')
    for _ in range(80):
        m = (a+b)/2
        if periodic(k, C, m): a = m
        else: b = m
    return a

sc = coeffs()
sc['Stormer explicit 3-step'] = sc.pop('Opt. explicit m=5,k=3')
out = {}
for name, (k, C) in sc.items():
    H0 = interval(k, C)
    out[name] = float(H0)
    print(f"{name:32s} periodicity interval (0, {mp.nstr(H0, 8)})")
json.dump(out, open('stab.json', 'w'), indent=1)

def absstable(k, C, H2, tol=1e-14):
    return all(abs(r) < 1 - tol for r in roots(k, C, H2))

def ainterval(k, C):
    H2 = mp.mpf('1e-4'); last = 0; first = None
    while H2 < 100:
        if absstable(k, C, H2):
            if first is None: first = H2
            last = H2
        elif first is not None: break
        H2 *= mp.mpf('1.01')
    if first is None: return None
    a, b = last, last*mp.mpf('1.01')
    for _ in range(80):
        m = (a+b)/2
        if absstable(k, C, m): a = m
        else: b = m
    return float(first), float(a)

print("--- absolute stability (all |zeta|<1) ---")
out2 = {}
for name, (k, C) in sc.items():
    r = ainterval(k, C); out2[name] = r
    print(f"{name:32s}", r)
    # print moduli at small H2 for information
    print("    moduli at H^2=0.01:", [mp.nstr(abs(x), 8) for x in roots(k, C, mp.mpf('0.01'))])
json.dump({'periodicity': out, 'absolute': out2}, open('stab.json', 'w'), indent=1)
