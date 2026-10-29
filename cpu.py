import time, numpy as np
from c2lib import *
e=0.5
def E_(t):
    E=t+e*np.sin(t)
    for _ in range(60): E=E-(E-e*np.sin(E)-t)/(1-e*np.cos(E))
    return E
q=lambda t:(lambda E: np.array([np.cos(E)-e,np.sqrt(1-e*e)*np.sin(E)]))(E_(t))
# cheap f (plain floats) to mimic compiled cost
def f(t,y):
    r2=y[0]*y[0]+y[1]*y[1]; s=r2**-1.5; return -y*s
P=100
def tm(fun):
    best=1e9
    for _ in range(3):
        t0=time.perf_counter(); Y=fun(); best=min(best,time.perf_counter()-t0)
    return best,Y
for epp in (800,1600,3200):
    print(f"-- e=0.5, {P} periods, {epp} f-evals/period")
    for nm in ('M6 [Liu et al., orbits]','NEW6 [Kovalnogov et al.]'):
        n=P*epp//4; h=P*2*np.pi/n
        t,Y=tm(lambda: hyb(HYB[nm],f,q,0,h,n)); ex=np.array([q(i*h) for i in range(0,n+1,50)])
        print(f"   {nm:26s} steps={n:7d} CPU={t:6.2f}s err={np.max(np.linalg.norm(Y[::50]-ex,axis=1)):.1e}")
    for nm in ('Stormer k=6','Opt. m=8 k=8'):
        n=P*epp; h=P*2*np.pi/n
        t,Y=tm(lambda: lmm(LMM[nm],f,q,0,h,n)); ex=np.array([q(i*h) for i in range(0,n+1,50)])
        print(f"   {nm:26s} steps={n:7d} CPU={t:6.2f}s err={np.max(np.linalg.norm(Y[::50]-ex,axis=1)):.1e}")
