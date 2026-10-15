# Rationalize the numerical SDP solution into an exact certificate file.
import pickle, json, numpy as np, itertools, math, collections
from fractions import Fraction as Fr
import mpmath as mp
from sdp import D, n, vi, canon, orbs, idx, corbs
mp.mp.dps=80
import sys
t,K,c=pickle.load(open(sys.argv[1],"rb"))
U=Fr(sys.argv[2]); U4=U**4
SK=50; SL=45; G0=Fr(1,2**int(sys.argv[3]))
# high-precision h and F orbit coefficients
hh=[mp.mpf(2)/n*mp.fsum(mp.sin(2*mp.pi*k*j/n) for k in range(1,16)) for j in range(n)]
def Fco(A):
    cnt=collections.Counter(A); mult=math.factorial(4)//math.prod(math.factorial(m) for m in cnt.values())
    return mult*mp.fsum(mp.fprod(hh[(x-j)%n]**m for j,m in cnt.items()) for x in range(n))
Fm=[Fco(A) for A in orbs]
# round c, K
ci=[int(round(v*2**SK)) for v in c]
Ki=np.zeros(D*D*n,dtype=object)
done=set()
for d in range(D):
  for e in range(D):
    for s in range(n):
      a,b=vi(d,e,s),vi(e,d,-s)
      v=int(round((K[a]+K[b])/2*2**SK)); Ki[a]=v; Ki[b]=v
# orbit sums
Sorb=[0]*len(orbs); members=collections.defaultdict(list)
for d in range(D):
  for e in range(D):
    for s in range(n):
      A=idx[canon((0,d,s,s+e))]; Sorb[A]+=Ki[vi(d,e,s)]; members[A].append((d,e,s))
Corb=[0]*len(orbs)
for bi,B in enumerate(corbs):
    for r in range(n): Corb[idx[canon((r,)+B)]]+=ci[bi]
# absorb: target T_A = U4*[A=0000] - F_A - C_A ; scaled by 2^SK
mx=0
for A in range(len(orbs)):
    T=(mp.mpf(U4.numerator)/U4.denominator if orbs[A]==(0,0,0,0) else 0) - Fm[A]
    T=T*2**SK - Corb[A]
    r=T-Sorb[A]
    # pick designated entry
    d,e,s=members[A][0]
    a,b=vi(d,e,s),vi(e,d,-s)
    if a==b: delta=int(mp.nint(r))
    else:
        delta=int(mp.nint(r/2))
    Ki[a]+=delta
    if a!=b: Ki[b]+=delta
    mx=max(mx,abs(float(r)))
print("max absorbed residual (units 2^-50):",mx)
# Gram matrix (exact ints, scale 2^SK) and Cholesky in float
N=D*n
Qi=np.zeros((N,N),dtype=object)
Kr=Ki.reshape(D,D,n)
for d in range(D):
  for e in range(D):
    for tt in range(n):
      for uu in range(n):
        Qi[d*n+tt,e*n+uu]=Kr[d,e,(uu-tt)%n]
Qf=Qi.astype(float)/2**SK
print("min eig",np.linalg.eigvalsh(Qf).min())
L=np.linalg.cholesky(Qf-float(G0)*np.eye(N))
Li=np.rint(L*2**SL).astype(np.int64)
# witness for lower bound
from op import H
from scipy.optimize import minimize
P0=np.ones((n,n))/n; B0=np.linalg.svd(np.eye(n)-P0)[0][:,:30]
def neg(x): f=B0@x; return -np.sum((H@f)**4)/np.sum(f**4)
rng=np.random.default_rng(1); best=None
for _ in range(200):
    r=minimize(neg,rng.standard_normal(30),method="BFGS",options={"gtol":1e-14})
    if best is None or r.fun<best.fun: best=r
f=B0@best.x; f/=np.abs(f).max()
qi=[int(round(v*2**40)) for v in f[:30]]; qi.append(-sum(qi))
json.dump({"U":[U.numerator,U.denominator],"gamma0":[G0.numerator,G0.denominator],
  "K_scale_log2":SK,"K":[int(v) for v in Ki],"cubic":[[list(B),ci[i]] for i,B in enumerate(corbs)],
  "L_scale_log2":SL,"L":[[int(v) for v in Li[i,:i+1]] for i in range(N)],
  "witness_scale_log2":40,"witness":qi}, open(sys.argv[4],"w"))
print("witness ratio",(-best.fun)**0.25)
