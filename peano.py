"""Second, independent check of ||l||^2 in L2^(m)* via the Peano kernel:
   l(phi) = int phi^(m)(t) Kl(t) dt,  Kl(t) = l_x[(x-t)_+^(m-1)/(m-1)!],
   ||l||^2 = int Kl(t)^2 dt  (h = 1, scales as h^(2m-1))."""
import sympy as sp
from opt import optimal
t = sp.symbols('t', real=True)

def norm2(m, k, Cpp):
    C = {k: 1, k-1: -2, k-2: 1}
    tot = 0
    for j in range(k):            # integrate piecewise on [j, j+1]
        expr = 0
        for b, c in C.items():
            if b > j:   # (b-t)_+ active for t in [j,j+1] iff b>=j+1
                expr += c*(b-t)**(m-1)/sp.factorial(m-1)
        for b, c in Cpp.items():
            if b > j and c != 0:
                expr -= c*(m-1)*(m-2)*(b-t)**(m-3)/sp.factorial(m-1)
        tot += sp.integrate(sp.expand(expr**2), (t, j, j+1))
    return sp.nsimplify(tot)

if __name__ == "__main__":
    cases = [(5, 3, False, {0: sp.Rational(1,12), 1: -sp.Rational(1,6), 2: sp.Rational(13,12), 3: 0}, "Thm5 (paper: 1/648 h^9)"),
             (6, 3, True, {0: 0, 1: sp.Rational(1,12), 2: sp.Rational(5,6), 3: sp.Rational(1,12)}, "Thm6 (paper: 61/3742200 h^11)"),
             (6, 4, False, {0: -sp.Rational(1,12), 1: sp.Rational(1,3), 2: -sp.Rational(5,12), 3: sp.Rational(7,6), 4: 0}, "Thm7 (paper: 27443/8553600 h^11)")]
    for m, k, imp, c, lab in cases:
        print(lab, "-> Peano:", norm2(m, k, c), f"h^{2*m-1}")
    for k in (3, 6):
        c, J, _ = optimal(4, k, True)
        print(f"m=4,k={k} implicit: KKT={J}  Peano={norm2(4, k, c)}  (h^7)")
    c, J, _ = optimal(4, 6, False)
    print(f"m=4,k=6 explicit: KKT={J}  Peano={norm2(4, 6, c)}  (h^7)")
