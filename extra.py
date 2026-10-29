"""Additional tests: oscillatory problem y''=-100y (y=cos 10x) and nonlinear y''=2y^3 (y=1/(1+x)) on [0,1].
Multistep schemes started with exact values; implicit nonlinear steps solved by Newton (30-digit mpmath)."""
import json, math, mpmath as mp
from fractions import Fraction as Fr
from num import coeffs
mp.mp.dps = 30
sc = coeffs()
P = {
 'osc': dict(f=lambda x, y: -100*y, ex=lambda x: mp.cos(10*x), y0=mp.mpf(1), v0=mp.mpf(0), fy=lambda x, y: mp.mpf(-100)),
 'nl':  dict(f=lambda x, y: 2*y**3, ex=lambda x: 1/(1+x), y0=mp.mpf(1), v0=mp.mpf(-1), fy=lambda x, y: 6*y**2),
}
def ms(pr, N, k, C):
    f, ex, fy = pr['f'], pr['ex'], pr['fy']
    h = mp.mpf(1)/N; x = [h*i for i in range(N+1)]
    ys = [ex(x[i]) for i in range(k)]
    Cm = {b: mp.mpf(c.numerator)/c.denominator if isinstance(c, Fr) else mp.mpf(c) for b, c in C.items()}
    for n in range(N-k+1):
        r = 2*ys[n+k-1] - ys[n+k-2] + h*h*sum(Cm[b]*f(x[n+b], ys[n+b]) for b in range(k))
        if Cm[k] == 0:
            y = r
        else:
            y = 2*ys[n+k-1] - ys[n+k-2]   # Newton for y - h^2 C[k] f(x,y) = r
            for _ in range(60):
                g = y - h*h*Cm[k]*f(x[n+k], y) - r
                dy = g/(1 - h*h*Cm[k]*fy(x[n+k], y)); y -= dy
                if abs(dy) < mp.mpf(10)**(-28): break
        ys.append(y)
    return float(max(abs(ys[i]-ex(x[i])) for i in range(N+1)))
def rk4(pr, N):
    f, ex = pr['f'], pr['ex']; h = mp.mpf(1)/N; y, v = pr['y0'], pr['v0']; e = 0
    for i in range(N):
        xi = h*i
        k1y, k1v = v, f(xi, y); k2y, k2v = v+h/2*k1v, f(xi+h/2, y+h/2*k1y)
        k3y, k3v = v+h/2*k2v, f(xi+h/2, y+h/2*k2y); k4y, k4v = v+h*k3v, f(xi+h, y+h*k3y)
        y = y+h/6*(k1y+2*k2y+2*k3y+k4y); v = v+h/6*(k1v+2*k2v+2*k3v+k4v); e = max(e, abs(y-ex(xi+h)))
    return float(e)
names = ['Verlet (Stormer, 2nd order)', 'Opt. implicit m=4,k=6', 'Opt. explicit m=5,k=3', 'Numerov', 'Opt. explicit m=6,k=4']
Ns = [20, 40, 80, 160, 320]
R = {}
for pn, pr in P.items():
    R[pn] = {}
    for nm in names:
        k, C = sc[nm]; R[pn][nm] = [ms(pr, N, k, C) for N in Ns]
    R[pn]['RK4'] = [rk4(pr, N) for N in Ns]
    print('==', pn)
    for nm, vs in R[pn].items():
        print(f"{nm:30s}", ' '.join(f"{v:.3e}" for v in vs), ' | ', ' '.join(f"{math.log2(vs[i-1]/vs[i]):.2f}" for i in range(1, len(vs))))
json.dump({'N': Ns, 'res': R}, open('extra.json', 'w'), indent=1)
