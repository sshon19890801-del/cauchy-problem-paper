import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from _lib import optimal, maximal_order
from family6 import build
HYB={nm:build(*p)[0] for nm,p in [('T6 [Tsitouras]',(0.5,-0.5)),('F6 [Franco]',(-0.2,-0.4)),('M6 [Liu et al., orbits]',(3/44,-23/38)),('NEW6 [Kovalnogov et al.]',(40/53,-37/60))]}
fl=lambda v: np.array([float(x) for x in v])
LMM={'Stormer k=6':fl(maximal_order(6,True)+[0]),'Opt. m=8 k=8':fl(optimal(8,8,True)[0]+[0])}
def hyb(M,f,yex,t0,h,n):
    a,D,w=M; Y=np.empty((n+1,)+np.shape(yex(t0))); Y[0]=yex(t0); Y[1]=yex(t0+h)
    fprev=f(t0,Y[0])
    for k in range(1,n):
        tk=t0+k*h; Fs=[fprev,f(tk,Y[k])]
        for i in range(2,5):
            v=(1+a[i])*Y[k]-a[i]*Y[k-1]+h*h*sum(D[i,j]*Fs[j] for j in range(i))
            Fs.append(f(tk+a[i]*h,v))
        Y[k+1]=2*Y[k]-Y[k-1]+h*h*sum(w[i]*Fs[i] for i in range(5)); fprev=Fs[1]
    return Y
def lmm(c,f,yex,t0,h,n):
    k=len(c)-1; Y=np.empty((n+1,)+np.shape(yex(t0)))
    for j in range(k): Y[j]=yex(t0+j*h)
    Fv=[f(t0+j*h,Y[j]) for j in range(k)]
    for m in range(k,n+1):
        Y[m]=2*Y[m-1]-Y[m-2]+h*h*sum(c[b]*Fv[m-k+b] for b in range(k)); Fv.append(f(t0+m*h,Y[m]))
    return Y
def run(name,f,yex,Tend,budgets,errf):
    print(f"\n{name}")
    print("   f-evals | "+" | ".join(f"{b:>9d}" for b in budgets))
    rows={}
    for nm,M in HYB.items():
        r=[]
        for B in budgets:
            n=B//4; h=Tend/n; Y=hyb(M,f,yex,0.0,h,n); r.append(errf(Y,h,n))
        rows[nm]=r
    for nm,c in LMM.items():
        r=[]
        for B in budgets:
            n=B; h=Tend/n; Y=lmm(c,f,yex,0.0,h,n); r.append(errf(Y,h,n))
        rows[nm]=r
    for nm,r in rows.items(): print(f"   {nm:26s}"+" ".join(f"{x:10.2e}" for x in r))
    return rows
# Kepler
def kepler(e):
    def E_(t):
        E=t+e*np.sin(t)
        for _ in range(60): E=E-(E-e*np.sin(E)-t)/(1-e*np.cos(E))
        return E
    q=lambda t:(lambda E: np.array([np.cos(E)-e,np.sqrt(1-e*e)*np.sin(E)]))(E_(t))
    f=lambda t,y: -y/np.sqrt(y[0]**2+y[1]**2)**3
    def err(Y,h,n):
        if not np.all(np.isfinite(Y)): return np.inf
        ex=np.array([q(i*h) for i in range(n+1)]); return np.max(np.linalg.norm(Y-ex,axis=1))
    return f,q,err
R={}
for e in (0.1,0.5,0.7):
    for P in (10,100):
        f,q,err=kepler(e)
        R[(e,P)]=run(f"Kepler e={e}, {P} periods, max position error", f,q,P*2*np.pi,[P*800,P*1600,P*3200],err)
# oscillatory problems of Kovalnogov et al. on [0,10pi]
def osc(nu):
    return (lambda t,y:-nu*nu*y),(lambda t: np.array([np.cos(nu*t)])),(lambda Y,h,n: np.max(np.abs(Y[:,0]-np.cos(nu*np.arange(n+1)*h))))
f,q,err=osc(5.0); R['osc5']=run("y''=-25y on [0,10pi]",f,q,10*np.pi,[2000,4000,8000],err)
f=lambda t,y:-100*y+99*np.sin(t); q=lambda t: np.array([np.cos(10*t)+np.sin(10*t)+np.sin(t)])
err=lambda Y,h,n: np.max(np.abs(Y[:,0]-np.array([q(i*h)[0] for i in range(n+1)])))
R['inh']=run("y''=-100y+99 sin t on [0,10pi]",f,q,10*np.pi,[4000,8000,16000],err)
