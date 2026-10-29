from fractions import Fraction as F
from math import factorial, sqrt, comb
import numpy as np
def G(m, d, x):
    p = 2 * m - 1 - d
    return F(abs(x) ** p, 2 * factorial(p))


def gauss(A, b):
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for i in range(n):
        p = next(r for r in range(i, n) if M[r][i] != 0)
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i and M[r][i] != 0:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


def norm2(m, C, c):
    s = (-1) ** m
    K, n = len(C), len(c)
    a = sum(C[i] * C[j] * G(m, 0, i - j) for i in range(K) for j in range(K))
    b = sum(C[i] * c[j] * G(m, 2, i - j) for i in range(K) for j in range(n))
    d = sum(c[i] * c[j] * G(m, 4, i - j) for i in range(n) for j in range(n))
    return s * (a - 2 * b + d)


def Cvec(k):
    C = [0] * (k + 1)
    C[k - 2], C[k - 1], C[k] = 1, -2, 1
    return C


def optimal(m, k, explicit):
    C = Cvec(k)
    n = k if explicit else k + 1
    s = (-1) ** m
    bv = [sum(C[i] * G(m, 2, i - g) for i in range(k + 1)) for g in range(n)]
    H = [[G(m, 4, i - j) for j in range(n)] for i in range(n)]
    rows = [[F(b) ** (a - 2) for b in range(n)] for a in range(2, m)]
    rhs = [F(sum(C[b] * b ** a for b in range(k + 1)), a * (a - 1)) for a in range(2, m)]
    nc = len(rows)
    A = [[2 * s * H[i][j] for j in range(n)] + [-rows[l][i] for l in range(nc)] for i in range(n)]
    A += [rows[l] + [F(0)] * nc for l in range(nc)]
    x = gauss(A, [2 * s * bb for bb in bv] + rhs)
    c = x[:n]
    return c, norm2(m, C, c)


def maximal_order(k, explicit):
    """classical Stormer (explicit) / Cowell (implicit) k-step scheme: exactness of maximal degree"""
    C = Cvec(k)
    n = k if explicit else k + 1
    rows = [[F(b) ** (a - 2) for b in range(n)] for a in range(2, n + 2)]
    rhs = [F(sum(C[b] * b ** a for b in range(k + 1)), a * (a - 1)) for a in range(2, n + 2)]
    return gauss(rows, rhs)


