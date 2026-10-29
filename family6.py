"""Sixth-order four-stage explicit Numerov-type family of Tsitouras type,
parametrized by a3,a4 (nodes a=(-1,0,a3,a4,a5)); conditions identified from NEW6."""
import numpy as np
def mom(p): return 0.0 if p%2 else 2/((p+1)*(p+2))
def build(a3,a4):
    def wfor(a5):
        a=np.array([-1,0,a3,a4,a5]); V=np.array([a**p for p in range(5)])
        return a, np.linalg.solve(V,[mom(p) for p in range(5)])
    g=lambda a5: (lambda a,w: w@a**5)(*wfor(a5))
    # find root of g by scanning
    xs=np.linspace(-1.5,1.5,30000)+1.2345e-5
    def gg(x):
        try: return g(x)
        except np.linalg.LinAlgError: return np.nan
    gs=[gg(x) for x in xs]; roots=[]
    for i in range(len(xs)-1):
        if np.isfinite(gs[i]) and np.isfinite(gs[i+1]) and gs[i]*gs[i+1]<0:
            lo,hi=xs[i],xs[i+1]
            for _ in range(100):
                mid=(lo+hi)/2
                if gg(lo)*gg(mid)<=0: hi=mid
                else: lo=mid
            if abs(g(lo))<1e-10: roots.append(lo)
    out=[]
    for a5 in roots:
        a,w=wfor(a5)
        D=np.zeros((5,5))
        # simplifying: D.1=(a^2+a)/2, D.a=(a^3-a)/6 ; free: d43,d53,d54
        def fill(d43,d53,d54):
            M=np.zeros((5,5))
            for i,(known) in [(2,{}),(3,{2:d43}),(4,{2:d53,3:d54})]:
                r1=(a[i]**2+a[i])/2; r2=(a[i]**3-a[i])/6
                for j,v in known.items(): r1-=v; r2-=v*a[j]
                # unknown d_i0 (a=-1), d_i1 (a=0): d0+d1=r1, -d0=r2
                M[i,0]=-r2; M[i,1]=r1+r2
                for j,v in known.items(): M[i,j]=v
            return M
        def conds(x):
            M=fill(*x); a2=a**2
            return np.array([w@M@a2-1/180, w@M@a**3, w@(a*(M@a2))-1/72])
        x=np.zeros(3); J=np.zeros((3,3)); c0=conds(x)
        for j in range(3):
            e=np.zeros(3); e[j]=1; J[:,j]=conds(e)-c0
        x=np.linalg.solve(J,-c0)
        out.append((a,fill(*x),w))
    return out
if __name__=="__main__":
    for a,D,w in build(40/53,-37/60):
        print(a); print(w); print(D)
