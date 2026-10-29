Python scripts reproducing all computations in the paper (Python 3, sympy, mpmath, matplotlib).
  opt.py            exact (rational) solution of the constrained minimisation -> optimal coefficients and norms
  peano.py          independent check of the norms via the Peano kernel
  verify_proofs.py  checks of the intermediate formulas of Theorems 3-4 and of the special cases (p_i)
  num.py, final.py  numerical experiments for y'' = x + y (Tables 1-2, final.json)
  extra.py          oscillatory and nonlinear test problems (Table 3, extra.json)
  stab.py           intervals of periodicity / absolute stability (Table 4, stab.json)
  fig.py            Figure 1
  revision_checks.py  checks added in the revision: simplified form of Theorem 4 (k=3..10), structure of
                    Theorems 1-2 for m=5,6 (Table struct), symmetric family (Proposition 1), norms in L_2^(4)*
                    of classical schemes (Table L24), work-precision data and figure2.png
                    (numpy, matplotlib only; output in revision_checks.out / revision_checks.json)
  _lib.py           shared exact-arithmetic routines (optimal and classical coefficients, norms)
  orbit_tests.py    Kepler problem and long-time oscillator for all schemes (Table kepler)
  promising.py      exact coefficients, norms, error constants, stability of the new schemes m=7 (k=5,6), m=8 (k=7) (Table newcoef)
  high_m.py, high_m_same_stencil.py, stability_same_stencil.out   schemes for m=8,9,10; same-stencil comparison; stability
  gain_vs_m.out     data of Table gain
  limited_smoothness.py, rough_fourier.py, underresolved.py   tests with solutions of limited smoothness (Discussion)

hybrid/  Comparison with trained hybrid Numerov-type methods (Table 7):
  family6.py   reconstruction of the sixth-order four-stage family from its free parameters a3, a4
  compare2.py  Kepler (e = 0.1, 0.5, 0.7) and oscillatory tests at equal numbers of f-evaluations; output in compare2.out
