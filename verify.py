#!/usr/bin/env python3
"""
Independent exact verifier for the P2730 certificate (Python standard library only).

Claim checked:  for every real f on Z/31Z with sum f = 0, f != 0,
    U^4 * sum_j f_j^4 - sum_j (Hf)_j^4  >  (mu - delta) * sum_j f_j^4  > 0,
and a rational witness q with sum q = 0 satisfies  F(q) > L^4 G(q).
Hence  L < C31 < U.

Argument (all steps checked below with exact rational arithmetic):
 1. Q (496x496, indexed by quadratic monomials z_{d,t} = f_t f_{t+d}, 0<=d<=15)
    is built from integer data K; Q is checked symmetric.
 2. Q - gamma0*I = L L^T + R exactly; ||R||_2 <= max row sum rho, so Q >= mu*I, mu = gamma0 - rho > 0.
    Then z^T Q z >= mu * sum z^2 >= mu * sum_t f_t^4   (d=0 monomials).
 3. P_U - z^T Q z - (sum_j f_j) * C(f) = sum_alpha e_alpha f^alpha  (monomial expansion, C a given cubic).
    The (Hf)^4 coefficients use rigorous rational enclosures of h(j); their error is bounded explicitly.
    |f^alpha| <= max_j f_j^4 <= G(f) (AM-GM), so |sum e f^alpha| <= delta*G with delta = sum|e| + error bound.
 4. On sum f = 0 the cubic term vanishes, so P_U >= (mu - delta) G.
"""
import json, sys, math
from fractions import Fraction as Fr
from collections import defaultdict

n = 31
GRID_S = 300          # bits for sine enclosures
GRID_H = 200          # bits for h midpoints

def round_grid(x, bits):
    """Round Fraction x to nearest multiple of 2^-bits; returns (int m, error <= 2^-bits)."""
    m = round(x * (1 << bits))
    return m

# ---------- rigorous pi ----------
def arctan_inv(k, terms):
    s = Fr(0)
    for i in range(terms):
        s += Fr((-1) ** i, (2 * i + 1) * k ** (2 * i + 1))
    err = Fr(1, (2 * terms + 1) * k ** (2 * terms + 1))   # alternating series bound
    return s, err

a5, e5 = arctan_inv(5, 200)
a239, e239 = arctan_inv(239, 80)
pi_mid_exact = 16 * a5 - 4 * a239
pi_err = 16 * e5 + 4 * e239
pm = Fr(round_grid(pi_mid_exact, GRID_S), 1 << GRID_S)
pi_rad = pi_err + Fr(1, 1 << GRID_S)
assert Fr(314159, 100000) < pm < Fr(314160, 100000)

# ---------- rigorous sin(2 pi m / 31) ----------
def sin_series(x, terms):
    s = Fr(0); term = x; k = 1
    for i in range(terms):
        s += term if i % 2 == 0 else -term
        term = term * x * x / ((k + 1) * (k + 2)); k += 2
    rem = abs(term)          # next term bounds the alternating remainder once terms decrease
    return s, rem

S_mid, S_err = [], []
for m in range(n):
    xm = 2 * pm * m / n
    xrad = 2 * pi_rad * m / n                # |x - xm|; sin is 1-Lipschitz
    s, rem = sin_series(xm, 120)
    assert rem < Fr(1, 10 ** 100)
    mi = round_grid(s, GRID_S)
    S_mid.append(Fr(mi, 1 << GRID_S))
    S_err.append(rem + xrad + Fr(1, 1 << GRID_S))
eps_s = max(S_err)

# ---------- h(j) = (2/31) sum_{k=1}^{15} sin(2 pi k j/31) ----------
h_int, eps_h = [], Fr(0)
for j in range(n):
    v = Fr(2, n) * sum(S_mid[(k * j) % n] for k in range(1, 16))
    e = Fr(2, n) * 15 * eps_s
    hi = round_grid(v, GRID_H)
    h_int.append(hi)
    eps_h = max(eps_h, e + Fr(1, 1 << GRID_H))
assert h_int[0] == 0 or abs(Fr(h_int[0], 1 << GRID_H)) <= eps_h
H1 = 1 << GRID_H
for j in range(n):
    assert abs(Fr(h_int[j], H1)) + eps_h < 1, "need |h|,|h~| < 1"
print(f"h enclosure radius < 2^{math.floor(math.log2(float(eps_h)))}")

# ---------- load certificate ----------
cert = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "certificate.json"))
U = Fr(*cert["U"]); U4 = U ** 4
gamma0 = Fr(*cert["gamma0"])
SK = cert["K_scale_log2"]; Kflat = cert["K"]
D = 16
assert len(Kflat) == D * D * n
def Kv(d, e, s): return Kflat[(d * D + e) * n + (s % n)]

# ---------- step 1: Gram matrix ----------
N = D * n
Q = [[Kv(d, e, u - t) for e in range(D) for u in range(n)] for d in range(D) for t in range(n)]
for i in range(N):
    for j in range(i):
        assert Q[i][j] == Q[j][i], "Q not symmetric"
print("Q symmetric (496x496)")

# ---------- step 2: exact PSD via L L^T + R ----------
SL = cert["L_scale_log2"]; L = cert["L"]
assert len(L) == N and all(len(L[i]) == i + 1 for i in range(N))
# R * 2^(2SL) = Q*2^(2SL-SK) - gamma0*2^(2SL) I - L L^T
assert 2 * SL >= SK
g0s = gamma0 * (1 << (2 * SL))
assert g0s.denominator == 1
g0s = g0s.numerator
shift = 2 * SL - SK
rho_num = 0
for i in range(N):
    Li = L[i]
    row = 0
    for j in range(N):
        Lj = L[j]
        m = min(i, j) + 1
        llt = sum(Li[k] * Lj[k] for k in range(m))
        r = (Q[i][j] << shift) - (g0s if i == j else 0) - llt
        row += abs(r)
    rho_num = max(rho_num, row)
rho = Fr(rho_num, 1 << (2 * SL))
mu = gamma0 - rho
assert mu > 0
print(f"rho = {float(rho):.3e},  mu = gamma0 - rho = {float(mu):.6e} > 0")

# ---------- step 3: monomial expansion ----------
def key(*ix): return tuple(sorted(x % n for x in ix))
SQ = defaultdict(int)                         # scale 2^SK
for d in range(D):
    for t in range(n):
        row = Q[d * n + t]
        for e in range(D):
            for u in range(n):
                v = row[e * n + u]
                if v: SQ[key(t, t + d, u, u + e)] += v
CB = defaultdict(int)                         # (sum f) * sum_B c_B sum_t prod f_{t+b}, scale 2^SK
for B, cv in cert["cubic"]:
    if cv == 0: continue
    for t in range(n):
        for r in range(n):
            CB[key(r, t + B[0], t + B[1], t + B[2])] += cv

import itertools
total_e = Fr(0); err_F = Fr(0)
HS = 1 << (4 * GRID_H)
for mono in itertools.combinations_with_replacement(range(n), 4):
    cnt = {}
    for j in mono: cnt[j] = cnt.get(j, 0) + 1
    mult = 24 // math.prod(math.factorial(m) for m in cnt.values())
    fm = 0
    for x in range(n):
        p = 1
        for j in mono: p *= h_int[(x - j) % n]
        fm += p
    Fmid = Fr(mult * fm, HS)
    Gc = 1 if len(cnt) == 1 else 0
    e = U4 * Gc - Fmid - Fr(SQ.get(mono, 0) + CB.get(mono, 0), 1 << SK)
    total_e += abs(e)
    err_F += mult * n * 4 * eps_h              # |prod h - prod h~| <= 4 max|h - h~| when |h|,|h~|<1
extra = set(SQ) | set(CB)
assert all(len(k) == 4 for k in extra)
delta = total_e + err_F
print(f"delta = sum|e| + trig error = {float(total_e):.3e} + {float(err_F):.1e}")
margin = mu - delta
assert margin > 0
print(f"mu - delta = {float(margin):.6e} > 0  =>  C31 < U = {U} = {float(U):.10f}")

# ---------- lower bound witness ----------
q = cert["witness"]
assert len(q) == n and sum(q) == 0
Lb = Fr(15693754353, 10 ** 10)
Gq = sum(Fr(v) ** 4 for v in q)
Flo = Fr(0)
for x in range(n):
    mid = sum(Fr(h_int[(x - y) % n], H1) * q[y] for y in range(n))
    rad = eps_h * sum(abs(v) for v in q)
    lo = abs(mid) - rad
    assert lo > 0
    Flo += lo ** 4
assert Flo > Lb ** 4 * Gq
# best 16-digit decimal lower bound certified by this witness
r4 = Flo / Gq
ND = 16
m = math.isqrt(math.isqrt(int(r4 * 10 ** (4 * ND))))           # floor(r4^(1/4) * 10^ND)
lo_dec = Fr(m, 10 ** ND)
assert lo_dec ** 4 < r4
lo_str = f"{m // 10 ** ND}.{m % 10 ** ND:0{ND}d}"
print(f"witness: F(q) > ({lo_str})^4 G(q)")
print(f"\nVERIFIED:  {lo_str} < C31 < {U.numerator}/{U.denominator} (= {float(U):.10f})")
