"""Independent construction of optimal difference formulas
   phi[k]-2phi[k-1]+phi[k-2] = h^2 sum_{b=0}^k Cpp[b] phi''[b]
in L2^(m)(0,1) by minimising the norm of the error functional
   l(x) = sum C[b] delta(x-hb) - h^2 sum Cpp[b] delta''(x-hb)
subject to exactness on polynomials of degree <= m-1 (KKT system).
Exact rational arithmetic (h = 1; ||l||^2 scales as h^(2m-3))."""
import sympy as sp

def K(m, x, d=0):
    # Green kernel G_m(x) = (-1)^m |x|^(2m-1) / (2 (2m-1)!) and its even derivatives
    p = 2*m - 1 - d
    return sp.Integer(-1)**m * sp.Abs(sp.Integer(x))**p / (2*sp.factorial(p))

def optimal(m, k, implicit=True):
    C = {k: 1, k-1: -2, k-2: 1}
    nodes = list(range(k+1)) if implicit else list(range(k))
    n = len(nodes)
    # quadratic form in Cpp: J = A0 - 2 b.c + c^T M c
    M = sp.zeros(n, n); bvec = sp.zeros(n, 1)
    for i, bi in enumerate(nodes):
        for j, gj in enumerate(nodes):
            M[i, j] = K(m, bi-gj, 4)
        bvec[i] = sum(cv*K(m, bb-bi, 2) for bb, cv in C.items())
    A0 = sum(c1*c2*K(m, b1-b2) for b1, c1 in C.items() for b2, c2 in C.items())
    # exactness constraints: sum C b^a - sum Cpp a(a-1) b^(a-2) = 0, a=2..m-1
    rows, rhs = [], []
    for a in range(2, m):
        rows.append([a*(a-1)*sp.Integer(b)**(a-2) for b in nodes])
        rhs.append(sum(cv*sp.Integer(bb)**a for bb, cv in C.items()))
    E = sp.Matrix(rows); r = sp.Matrix(rhs)
    p = E.shape[0]
    KKT = sp.zeros(n+p, n+p)
    KKT[:n, :n] = 2*M; KKT[:n, n:] = E.T; KKT[n:, :n] = E
    sol = KKT.LUsolve(sp.Matrix.vstack(2*bvec, r))
    c = sol[:n, 0]
    J = sp.nsimplify(sp.simplify(A0 - 2*(bvec.T*c)[0] + (c.T*M*c)[0]))
    coeffs = {b: sp.nsimplify(sp.simplify(c[i])) for i, b in enumerate(nodes)}
    if not implicit:
        coeffs[k] = sp.Integer(0)
    return coeffs, J, 2*m-3

if __name__ == "__main__":
    q = sp.sqrt(3) - 2
    A2 = 6*sp.sqrt(3)*q
    def thm3(k):
        c = {0: (q**(k+1)-q**(k-1))/(5*(1-q**(2*k)))}
        for b in range(1, k-2):
            c[b] = 6*q**k*(q**(-b)-q**b)/(5*(1-q**(2*k)))
        c[k-2] = sp.Rational(1, 20) + 6*q**2*(1-q**(2*k-4))/(5*(1-q**(2*k)))
        c[k-1] = sp.Rational(11, 10) + 6*q*(1-q**(2*k-2))/(5*(1-q**(2*k)))
        c[k] = -sp.Rational(3, 20) + (1-q)*(1+q**(2*k-1))/(5*(1-q**(2*k)))
        return c
    def thm4(k):
        return (sp.Rational(11587, 12600) - q*(q**(2*k-2)-1)/(25*(1-q**(2*k)))
                - 6*q/(5*A2*(1-q**(2*k))) + (2*q-1)/(2*(1-q)) + 1/(20*q))
    print("== Theorem 3 / 4 check (m=4, implicit) ==")
    for k in (3, 4, 5, 6, 8):
        c, J, p = optimal(4, k, True)
        t = thm3(k)
        dc = max(abs(sp.N(c[b]-t[b], 30)) for b in c)
        print(f"k={k}: max|C_KKT - C_Thm3| = {sp.N(dc,3)};  ||l||^2/h^7: KKT={sp.N(J,12)}  Thm4={sp.N(thm4(k),12)}")
    c6, J6, _ = optimal(4, 6, True)
    print("m=4,k=6 implicit:", {b: str(v) for b, v in c6.items()}, "sum", sum(c6.values()))
    e6, Je6, _ = optimal(4, 6, False)
    print("m=4,k=6 explicit:", {b: str(v) for b, v in e6.items()}, " J=", Je6, sp.N(Je6))
    for (m, k, imp, label) in [(5, 3, False, "Thm5"), (6, 3, True, "Thm6"), (6, 4, False, "Thm7")]:
        c, J, p = optimal(m, k, imp)
        print(label, f"m={m},k={k},{'impl' if imp else 'expl'}:", {b: str(v) for b, v in c.items()}, f" ||l||^2 = {J} h^{p}")
