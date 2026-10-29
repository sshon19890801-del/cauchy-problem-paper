"""Numerical experiments for y'' = x + y, y(0)=2, y'(0)=-1 on [0,1], exact y = e^x + e^-x - x.
All k-step schemes  y[n+k]-2y[n+k-1]+y[n+k-2] = h^2 sum C[b] f[n+b];  starting values by RK4.
Run in double precision and in 40-digit arithmetic (mpmath) to separate truncation from round-off."""
import sys, json
from fractions import Fraction as Fr
import mpmath as mp

def coeffs():
    c = {}
    c['Verlet (Stormer, 2nd order)'] = (2, {0: 0, 1: 1, 2: 0})
    c['Numerov'] = (2, {0: Fr(1, 12), 1: Fr(10, 12), 2: Fr(1, 12)})
    c['Opt. explicit m=4,k=6'] = (6, {0: Fr(-1, 4180), 1: Fr(3, 2090), 2: Fr(-6, 1045), 3: Fr(9, 418), 4: Fr(-127, 4180), 5: Fr(1059, 1045), 6: 0})
    c['Opt. implicit m=4,k=6'] = (6, {0: Fr(1, 3900), 1: Fr(-1, 650), 2: Fr(2, 325), 3: Fr(-3, 130), 4: Fr(177, 1300), 5: Fr(253, 325), 6: Fr(101, 975)})
    c['Opt. explicit m=5,k=3'] = (3, {0: Fr(1, 12), 1: Fr(-1, 6), 2: Fr(13, 12), 3: 0})
    c['Opt. implicit m=6,k=3'] = (3, {0: 0, 1: Fr(1, 12), 2: Fr(5, 6), 3: Fr(1, 12)})
    c['Opt. explicit m=6,k=4'] = (4, {0: Fr(-1, 12), 1: Fr(1, 3), 2: Fr(-5, 12), 3: Fr(7, 6), 4: 0})
    return c

def run(N, num, kind, scheme=None):
    one = num(1)
    h = one / N
    x = [h*i for i in range(N+1)]
    exact = [num.exp(xi) + num.exp(-xi) - xi for xi in x] if hasattr(num, 'exp') else None
    f = lambda xx, yy: xx + yy
    def rk4(steps):
        y, v = num(2), num(-1); ys = [y]
        for i in range(steps):
            xi = x[i]
            k1y, k1v = v, f(xi, y)
            k2y, k2v = v + h/2*k1v, f(xi + h/2, y + h/2*k1y)
            k3y, k3v = v + h/2*k2v, f(xi + h/2, y + h/2*k2y)
            k4y, k4v = v + h*k3v, f(xi + h, y + h*k3y)
            y = y + h/6*(k1y + 2*k2y + 2*k3y + k4y)
            v = v + h/6*(k1v + 2*k2v + 2*k3v + k4v)
            ys.append(y)
        return ys
    if kind == 'RK4':
        ys = rk4(N)
    elif kind == 'RK2':          # explicit midpoint method on the first-order system
        y, v = num(2), num(-1); ys = [y]
        for i in range(N):
            k1y, k1v = v, f(x[i], y)
            y, v = y + h*(v + h/2*k1v), v + h*f(x[i] + h/2, y + h/2*k1y)
            ys.append(y)
    else:
        k, C = scheme
        C = {b: num(c.numerator)/num(c.denominator) if isinstance(c, Fr) else num(c) for b, c in C.items()}
        ys = rk4(k-1) if k > 1 else [num(2)]
        for n in range(0, N-k+1):
            rhs = 2*ys[n+k-1] - ys[n+k-2] + h*h*(sum(C[b]*f(x[n+b], ys[n+b]) for b in range(k)) + C[k]*x[n+k])
            ys.append(rhs / (1 - h*h*C[k]))
    return max(abs(ys[i] - exact[i]) for i in range(N+1))

class D:   # double precision namespace
    import math
    def __call__(self, v): return float(v)
    exp = staticmethod(math.exp)

if __name__ == "__main__":
    mp.mp.dps = 40
    MP = lambda v: mp.mpf(v); MP.exp = mp.exp
    dbl = D()
    sc = coeffs()
    out = {}
    t1N = [50, 100, 200, 400, 800]
    t2N = [10, 20, 40, 80, 160, 320]
    for prec, num in (('double', dbl), ('mp40', MP)):
        res = {}
        for name in ['Verlet (Stormer, 2nd order)', 'Opt. explicit m=4,k=6', 'Opt. implicit m=4,k=6', 'Numerov']:
            res[name] = [run(N, num, 'ms', sc[name]) for N in t1N]
        res['RK2'] = [run(N, num, 'RK2') for N in t1N]
        res['RK4'] = [run(N, num, 'RK4') for N in t1N]
        for name in ['Opt. explicit m=5,k=3', 'Opt. implicit m=6,k=3', 'Opt. explicit m=6,k=4']:
            res[name + ' T2'] = [run(N, num, 'ms', sc[name]) for N in t2N]
        out[prec] = {k: [float(v) for v in vs] for k, vs in res.items()}
    json.dump(out, open('results.json', 'w'), indent=1)
    import math
    for prec in out:
        print('=====', prec)
        for k, vs in out[prec].items():
            Ns = t2N if k.endswith('T2') else t1N
            orders = ['-'] + [f"{math.log2(vs[i-1]/vs[i]):.4f}" for i in range(1, len(vs))]
            print(f"{k:34s}", '  '.join(f"{N}:{v:.5e}({o})" for N, v, o in zip(Ns, vs, orders)))
