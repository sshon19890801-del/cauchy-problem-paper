import json, mpmath as mp
from fractions import Fraction as Fr
from num import coeffs
mp.mp.dps = 40
ex = lambda x: mp.exp(x)+mp.exp(-x)-x
f = lambda a,b: a+b
def rk4_vals(N, steps):
    h=mp.mpf(1)/N; y,v=mp.mpf(2),mp.mpf(-1); ys=[y]
    for i in range(steps):
        xi=h*i
        k1y,k1v=v,f(xi,y); k2y,k2v=v+h/2*k1v,f(xi+h/2,y+h/2*k1y)
        k3y,k3v=v+h/2*k2v,f(xi+h/2,y+h/2*k2y); k4y,k4v=v+h*k3v,f(xi+h,y+h*k3y)
        y=y+h/6*(k1y+2*k2y+2*k3y+k4y); v=v+h/6*(k1v+2*k2v+2*k3v+k4v); ys.append(y)
    return ys
def ms(N,k,C,start):
    h=mp.mpf(1)/N; x=[h*i for i in range(N+1)]
    if start=='exact': ys=[ex(x[i]) for i in range(k)]
    elif start=='rk4': ys=rk4_vals(N,k-1)
    elif start=='taylor2': ys=[mp.mpf(2), 2+h*(-1+h/2*f(0,mp.mpf(2)))]
    Cm={b:mp.mpf(c.numerator)/c.denominator if isinstance(c,Fr) else mp.mpf(c) for b,c in C.items()}
    for n in range(N-k+1):
        r=2*ys[n+k-1]-ys[n+k-2]+h*h*(sum(Cm[b]*f(x[n+b],ys[n+b]) for b in range(k))+Cm[k]*x[n+k])
        ys.append(r/(1-h*h*Cm[k]))
    return float(max(abs(ys[i]-ex(x[i])) for i in range(N+1)))
def rk2(N):
    h=mp.mpf(1)/N; y,v=mp.mpf(2),mp.mpf(-1); e=0
    for i in range(N):
        xi=h*i; k1v=f(xi,y); y,v=y+h*(v+h/2*k1v), v+h*f(xi+h/2,y+h/2*v); e=max(e,abs(y-ex(xi+h)))
    return float(e)
def rk4(N):
    ys=rk4_vals(N,N); h=mp.mpf(1)/N
    return float(max(abs(ys[i]-ex(h*i)) for i in range(N+1)))
sc=coeffs()
T1=[50,100,200,400,800]; T2=[10,20,40,80,160,320]
R={'T1':{},'T2':{}}
R['T1']['Verlet']=[ms(N,2,sc['Verlet (Stormer, 2nd order)'][1],'taylor2') for N in T1]
R['T1']['RK2']=[rk2(N) for N in T1]
k,C=sc['Opt. explicit m=4,k=6']; R['T1']['OptExp']=[ms(N,k,C,'rk4') for N in T1]
k,C=sc['Opt. implicit m=4,k=6']; R['T1']['OptImp']=[ms(N,k,C,'rk4') for N in T1]
R['T1']['RK4']=[rk4(N) for N in T1]
k,C=sc['Numerov']; R['T1']['Numerov']=[ms(N,k,C,'rk4') for N in T1]
for nm,key in [('Opt. explicit m=5,k=3','E53'),('Opt. implicit m=6,k=3','I63'),('Opt. explicit m=6,k=4','E64')]:
    k,C=sc[nm]; R['T2'][key]=[ms(N,k,C,'exact') for N in T2]
R['T2']['RK4']=[rk4(N) for N in T2]
json.dump(R,open('final.json','w'),indent=1)
import math
for t,Ns in (('T1',T1),('T2',T2)):
    for kk,vs in R[t].items():
        print(t,kk,' '.join(f"{v:.5e}" for v in vs),' | ',' '.join(f"{math.log2(vs[i-1]/vs[i]):.4f}" for i in range(1,len(vs))))
