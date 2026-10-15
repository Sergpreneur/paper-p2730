# Numerical SDP: P_U - (sum f)*C(f) = z^T Q z,  Q translation-invariant,  maximize block margin gamma
import numpy as np, cvxpy as cp, itertools, math, collections, scipy.sparse as sp
from op import h, n
D=16
def canon(ms): return min(tuple(sorted(((a+t)%n) for a in ms)) for t in range(n))
orbs=sorted({canon(c) for c in itertools.combinations_with_replacement(range(n),4)}); idx={o:i for i,o in enumerate(orbs)}
corbs=sorted({canon(c) for c in itertools.combinations_with_replacement(range(n),3)})
def Fcoef(A):
    c=collections.Counter(A); mult=math.factorial(4)/np.prod([math.factorial(m) for m in c.values()])
    return mult*sum(np.prod([h[(x-j)%n]**m for j,m in c.items()]) for x in range(n))
Fc=np.array([Fcoef(A) for A in orbs]); Gc=np.zeros(len(orbs)); Gc[idx[(0,0,0,0)]]=1
nv=D*D*n
def vi(d,e,s): return (d*D+e)*n+s%n
rows=[idx[canon((0,d,s,s+e))] for d in range(D) for e in range(D) for s in range(n)]
Amap=sp.csr_matrix((np.ones(nv),(rows,range(nv))),shape=(len(orbs),nv))
rr=[];cc=[]
for bi,B in enumerate(corbs):
    for r in range(n): rr.append(idx[canon((r,)+B)]); cc.append(bi)
Cmap=sp.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(orbs),len(corbs)))
def solve(U):
    K=cp.Variable(nv); c=cp.Variable(len(corbs)); g=cp.Variable()
    sr=[];sc=[];sv=[];r=0
    for d in range(D):
      for e in range(D):
        for s in range(n):
          a,b=vi(d,e,s),vi(e,d,-s)
          if a<b: sr+=[r,r]; sc+=[a,b]; sv+=[1,-1]; r+=1
    cons=[sp.csr_matrix((sv,(sr,sc)),shape=(r,nv))@K==0, Amap@K+Cmap@c==U**4*Gc-Fc]
    for k in range(16):
        Cs=np.cos(2*np.pi*k*np.arange(n)/n); Ss=-np.sin(2*np.pi*k*np.arange(n)/n)
        R=cp.reshape(sp.kron(sp.eye(D*D),Cs.reshape(1,-1))@K,(D,D),order='C')
        I_=cp.reshape(sp.kron(sp.eye(D*D),Ss.reshape(1,-1))@K,(D,D),order='C')
        M=cp.bmat([[R,-I_],[I_,R]]); cons.append((M+M.T)/2-g*np.eye(2*D)>>0)
    p=cp.Problem(cp.Maximize(g),cons); p.solve(solver="CLARABEL")
    return p.status,g.value,K.value,c.value
if __name__=="__main__":
    import sys
    for U in map(float,sys.argv[1:]):
        st,g,_,_=solve(U); print(U,st,g,flush=True)
