"""Reviewer checks of the m=4 proofs (Sections VIII) and special cases."""
import sympy as sp
from opt import optimal
b, k, h = sp.symbols('beta k h')
q = sp.sqrt(3) - 2
A2 = 6*sp.sqrt(3)*q

# --- z_2, f_2 (20)/(54): f_2[b] = sum_g C[g] G4''(b-g), G4'' = |x|^5/(2*5!)
C = {k: sp.Integer(1), k-1: sp.Integer(-2), k-2: sp.Integer(1)}
def f2(bb, kk):
    return sum(c.subs(k, kk)*sp.Abs(bb-g.subs(k, kk))**5/(2*sp.factorial(5)) for g, c in C.items())
for kk in (3, 5, 8):
    z2 = -sp.Rational(1, 24)*(2*b**3 - 6*b**2*(kk-1) + b*(6*(kk-1)**2+1) - (kk-1)*(2*(kk-1)**2+1))
    ok = all(sp.simplify(f2(bb, kk) - z2.subs(b, bb)) == 0 for bb in range(0, kk-1))
    print(f"k={kk}: z2 matches f2 on 0..k-2: {ok}; f2[k-1]={f2(kk-1,kk)} (claim 1/120), f2[k]={f2(kk,kk)} (claim 1/8)")
    # (66): z2 expanded
    e66 = -b**3/12 + (kk-1)*b**2/4 - (sp.Rational(kk**2, 4)-sp.Rational(kk, 2)-sp.Rational(7, 24))*b + sp.Rational(kk**3, 12)-sp.Rational(kk**2, 4)+sp.Rational(7*kk, 24)-sp.Rational(1, 8)
    print("   (66) == z2:", sp.simplify(e66 - z2.subs(b, b)) == 0)

# --- Theorem 3 vs KKT, and polynomial P1 = b0 + b1[beta] (Lagrange multipliers)
def thm3(kk):
    c = {0: (q**(kk+1)-q**(kk-1))/(5*(1-q**(2*kk)))}
    for bb in range(1, kk-2):
        c[bb] = 6*q**kk*(q**(-bb)-q**bb)/(5*(1-q**(2*kk)))
    c[kk-2] = sp.Rational(1, 20) + 6*q**2*(1-q**(2*kk-4))/(5*(1-q**(2*kk)))
    c[kk-1] = sp.Rational(11, 10) + 6*q*(1-q**(2*kk-2))/(5*(1-q**(2*kk)))
    c[kk] = -sp.Rational(3, 20) + (1-q)*(1+q**(2*kk-1))/(5*(1-q**(2*kk)))
    return c
for kk in (3, 4, 6):
    c = thm3(kk)
    # P1[beta] = f2[beta] - h^2 sum C G4^IV(b-g), G4^IV=|x|^3/12 ; h=1
    P = [sp.nsimplify(sp.simplify(f2(bb, kk) - sum(c[g]*sp.Abs(bb-g)**3/12 for g in c))) for bb in range(kk+1)]
    b1 = sp.simplify(P[1]-P[0]); b0 = sp.simplify(P[0])
    lin = all(sp.simplify(P[i]-b0-b1*i) == 0 for i in range(kk+1))
    b1c = sp.Rational(1, 2)*(6*q/(5*A2*(q**kk-1)) + (2*q-1)/(2*(1-q)) + 1/(20*q) + sp.Rational(11, 12))
    b0c = sp.Rational(kk, 2)*(6*q/(5*A2*(1-q**(2*kk))) + (2*q-1)/(2*(q-1)) - 1/(20*q) - sp.Rational(11, 12))
    print(f"k={kk}: P1 linear: {lin}; b1 actual {sp.N(b1,12)} claim {sp.N(b1c,12)}; b0 actual {sp.N(b0,12)} claim {sp.N(b0c,12)}")
    # (70): b0^- = (k-1)(2(k-1)^2+1)/24 ; b0^- = b0 + (1/12) sum C g^3
    b0m = b0 + sp.Rational(1, 12)*sum(c[g]*g**3 for g in c)
    print("   b0^- actual", sp.N(b0m, 12), "claim", sp.N(sp.Rational((kk-1)*(2*(kk-1)**2+1), 24), 12))
    b1m = b1 - sp.Rational(1, 4)*sum(c[g]*g**2 for g in c)
    b1mc = -(6*q**(kk+1)/(5*A2*(1-q**(2*kk))) + sp.Rational((kk-1)**2, 4) + sp.Rational(1, 24))
    print("   b1^- actual", sp.N(b1m, 12), "claim", sp.N(b1mc, 12))
    b1p = b1 + sp.Rational(1, 4)*sum(c[g]*g**2 for g in c)
    b1pc = -(6*q/(5*A2*(1-q**(2*kk))) + (1-2*q)/(2*(1-q)) - 1/(20*q) - sp.Rational(kk**2, 4) + sp.Rational(kk, 2) - sp.Rational(29, 24))
    print("   b1^+ actual", sp.N(b1p, 12), "claim", sp.N(b1pc, 12))
    b0p = b0 - sp.Rational(1, 12)*sum(c[g]*g**3 for g in c)
    b0pc = sp.Rational(1, 8) + sp.Rational(kk**2*(2*kk-3), 12) - b1p*kk
    print("   b0^+ actual", sp.N(b0p, 12), "claim (71)", sp.N(b0pc, 12))
    # m, n (68),(69) vs. fit
    m_ = -6*q**kk/(5*(1-q**(2*kk))); n_ = 6*q/(5*(1-q**(2*kk)))
    print("   C[k-2] = 1/20 + m q^{k-2} + n q:", sp.simplify(c[kk-2] - (sp.Rational(1, 20)+m_*q**(kk-2)+n_*q)) == 0,
          " with q^{k-1} (as printed):", sp.simplify(c[kk-2] - (sp.Rational(1, 20)+m_*q**(kk-1)+n_*q)) == 0)
    print("   C[k-1] = 11/10 + m q^{k-1} + n:", sp.simplify(c[kk-1] - (sp.Rational(11, 10)+m_*q**(kk-1)+n_)) == 0)
    # (57), (58)
    c57 = sp.Rational(4-3*kk, 20*kk) + (m_-n_)*(q**kk-q)/((1-q)**2*kk) + (m_*q**kk-n_)*(kk-1)/((1-q)*kk)
    c58 = -sp.Rational(1, 5*kk) + (m_-n_)*(q-q**kk)/(kk*(1-q)**2) + (m_*q**kk-n_)/(kk*(1-q)) - (m_*q-n_*q**(kk-1))/(1-q)
    print("   (57):", sp.simplify(c[kk]-c57) == 0, " (58):", sp.simplify(c[0]-c58) == 0)
    # T2 claim
    T2c = -(-sp.Rational(537, 600) + q*(q**(2*kk-2)-1)/(25*(1-q**(2*kk))) + 6*q/(5*A2*(1-q**(2*kk))) - (2*q-1)/(2*(1-q)) - 1/(20*q))
    T2 = -sum(c[g]*(f2(g, kk) + b0 + b1*g) for g in c)
    print("   T2 actual", sp.N(T2, 12), "claim", sp.N(T2c, 12), "; T1 = 31/1260 =", sp.N(sp.Rational(31, 1260), 12))

# --- special cases: p_i
def multipliers(m, kk, implicit):
    c, J, _ = optimal(m, kk, implicit)
    nodes = range(kk+1) if implicit else range(kk)
    G2 = lambda x: sp.Abs(x)**(2*m-3)/(2*sp.factorial(2*m-3))
    G4 = lambda x: sp.Abs(x)**(2*m-5)/(2*sp.factorial(2*m-5))
    P = [sp.nsimplify(sum(cc.subs(k, kk)*G2(bb-g.subs(k, kk)) for g, cc in C.items()) - sum(c[g]*G4(bb-g) for g in nodes)) for bb in nodes]
    xs = sp.symbols('x')
    poly = sp.interpolate(list(zip(range(len(P)), P)), xs)
    return c, sp.Poly(sp.expand(poly), xs).all_coeffs()[::-1], J
for m, kk, imp, lab in [(5, 3, False, "m=5,k=3 expl"), (6, 3, True, "m=6,k=3 impl"), (6, 4, False, "m=6,k=4 expl")]:
    c, p, J = multipliers(m, kk, imp)
    print(lab, "C:", c, " p:", p, " J:", J)
