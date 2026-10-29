"""Kepler two-body orbit and long-time harmonic oscillator for all schemes (numpy only)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction as F
from _lib import optimal, maximal_order

SCH = {}
SCH["Verlet (o2)"] = [0, 1, 0]
SCH["Sym. opt a=1/10 (o2)"] = [F(1, 10), F(8, 10), F(1, 10)]
SCH["Opt. impl m4 k6 (o2)"] = optimal(4, 6, False)[0]
SCH["Opt. expl m4 k6 (o2)"] = optimal(4, 6, True)[0] + [0]
SCH["Stormer3 Th5 (o3)"] = [F(1, 12), F(-2, 12), F(13, 12), 0]
SCH["Numerov (o4)"] = [F(1, 12), F(10, 12), F(1, 12)]
SCH["Stormer4 Th7 (o4)"] = [F(-1, 12), F(4, 12), F(-5, 12), F(14, 12), 0]
SCH["Opt. impl m6 k6 (o4)"] = optimal(6, 6, False)[0]
SCH["Opt. expl m6 k6 (o4)"] = optimal(6, 6, True)[0] + [0]
SCH["Stormer6 (o6)"] = maximal_order(6, True) + [0]
SCH["Cowell6 (o7)"] = maximal_order(6, False)
SCH = {k: np.array([float(x) for x in v]) for k, v in SCH.items()}


def lmm(c, f, yex, h, nsteps):
    k = len(c) - 1
    t = np.arange(nsteps + 1) * h
    Y = np.empty((nsteps + 1,) + np.shape(yex(0.0)))
    for j in range(k):
        Y[j] = yex(t[j])
    Fv = [f(Y[j]) for j in range(k)]
    nev = 0
    for n in range(k, nsteps + 1):
        base = 2 * Y[n - 1] - Y[n - 2]
        rhs = base + h * h * sum(c[b] * Fv[n - k + b] for b in range(k))
        if c[k] == 0:
            Y[n] = rhs
        else:
            y = 3 * Y[n - 1] - 3 * Y[n - 2] + Y[n - 3]
            for it in range(50):
                ynew = rhs + h * h * c[k] * f(y); nev += 1
                if np.max(np.abs(ynew - y)) < 1e-15 * (1 + np.max(np.abs(y))):
                    y = ynew; break
                y = ynew
            Y[n] = y
        Fv.append(f(Y[n])); nev += 1
    return t, Y, nev


def rk4(f, yex, vex, h, nsteps):
    z = np.concatenate([yex(0.0), vex(0.0)]); d = len(z) // 2
    g = lambda z: np.concatenate([z[d:], f(z[:d])])
    Y = [z[:d].copy()]
    for _ in range(nsteps):
        k1 = g(z); k2 = g(z + h / 2 * k1); k3 = g(z + h / 2 * k2); k4 = g(z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); Y.append(z[:d].copy())
    return np.arange(nsteps + 1) * h, np.array(Y), 4 * nsteps


# ---- Kepler problem, eccentricity e, period 2*pi
e = 0.5
def kep_f(q):
    r = np.sqrt(q[0] ** 2 + q[1] ** 2); return -q / r ** 3
def ecc_anom(t):
    E = t + e * np.sin(t)
    for _ in range(60):
        E = E - (E - e * np.sin(E) - t) / (1 - e * np.cos(E))
    return E
def kep_q(t):
    E = ecc_anom(t); return np.array([np.cos(E) - e, np.sqrt(1 - e * e) * np.sin(E)])
def kep_v(t):
    E = ecc_anom(t); Ed = 1 / (1 - e * np.cos(E)); return np.array([-np.sin(E) * Ed, np.sqrt(1 - e * e) * np.cos(E) * Ed])

def energy_from_positions(Y, h):
    # 8th-order central difference for velocity at interior points
    w = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
    V = sum(w[i] * Y[i:len(Y) - 8 + i] for i in range(9)) / h
    Q = Y[4:-4]
    return 0.5 * np.sum(V ** 2, axis=1) - 1 / np.sqrt(np.sum(Q ** 2, axis=1))

res = {"kepler": {}, "oscillator": {}}
T = 2 * np.pi
print(f"KEPLER e={e}: max position error over 10 periods / 100 periods, energy error, steps per period")
for spp in (200, 400):
    h = T / spp
    print(f"--- {spp} steps per period (h={h:.4f})")
    for name, c in list(SCH.items()) + [("RK4 (o4)", None)]:
        out = {}
        for P in (10, 100):
            n = P * spp
            if c is None:
                t, Y, nev = rk4(kep_f, kep_q, kep_v, h, n)
            else:
                t, Y, nev = lmm(c, kep_f, kep_q, h, n)
            if not np.all(np.isfinite(Y)):
                out[P] = (np.inf, np.inf, nev); continue
            ex = np.array([kep_q(tt) for tt in t])
            err = np.max(np.linalg.norm(Y - ex, axis=1))
            H = energy_from_positions(Y, h)
            out[P] = (err, np.max(np.abs(H + 0.5)), nev)
        res["kepler"].setdefault(name, {})[spp] = out
        print(f"  {name:22s} err10={out[10][0]:9.2e} err100={out[100][0]:9.2e} dH100={out[100][1]:9.2e} f-evals/step={out[100][2]/(100*spp):.2f}")

# ---- harmonic oscillator y'' = -w^2 y, w = 10, long time
w = 10.0
osc_f = lambda y: -w * w * y
osc_y = lambda t: np.array([np.cos(w * t)])
osc_v = lambda t: np.array([-w * np.sin(w * t)])
print(f"OSCILLATOR w={w}: max error on [0,Tend], h=1/80")
h = 1 / 80
for Tend in (10, 100):
    n = int(round(Tend / h))
    for name, c in list(SCH.items()) + [("RK4 (o4)", None)]:
        if c is None:
            t, Y, _ = rk4(osc_f, osc_y, osc_v, h, n)
        else:
            t, Y, _ = lmm(c, osc_f, osc_y, h, n)
        err = np.max(np.abs(Y[:, 0] - np.cos(w * t)))
        amp = np.max(np.abs(Y[-int(2 * np.pi / w / h) - 1:, 0]))
        res["oscillator"].setdefault(name, {})[Tend] = (float(err), float(amp))
        print(f"  T={Tend:4d} {name:22s} err={err:9.2e}  final amplitude={amp:.6f}")

json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_tests.json"), "w"), indent=1, default=float)
