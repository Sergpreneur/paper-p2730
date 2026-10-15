from sdp import *
import pickle, sys
def solve_t(greq):
    K=cp.Variable(nv); c=cp.Variable(len(corbs)); t=cp.Variable()
    sr=[];sc=[];sv=[];r=0
    for d in range(D):
      for e in range(D):
        for s in range(n):
          a,b=vi(d,e,s),vi(e,d,-s)
          if a<b: sr+=[r,r]; sc+=[a,b]; sv+=[1,-1]; r+=1
    cons=[sp.csr_matrix((sv,(sr,sc)),shape=(r,nv))@K==0, Amap@K+Cmap@c==t*Gc-Fc]
    for k in range(16):
        Cs=np.cos(2*np.pi*k*np.arange(n)/n); Ss=-np.sin(2*np.pi*k*np.arange(n)/n)
        R=cp.reshape(sp.kron(sp.eye(D*D),Cs.reshape(1,-1))@K,(D,D),order='C')
        I_=cp.reshape(sp.kron(sp.eye(D*D),Ss.reshape(1,-1))@K,(D,D),order='C')
        M=cp.bmat([[R,-I_],[I_,R]]); cons.append((M+M.T)/2-greq*np.eye(2*D)>>0)
    p=cp.Problem(cp.Minimize(t),cons)
    p.solve(solver="CLARABEL",tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=500)
    return p.status,t.value,K.value,c.value
for greq in map(float,sys.argv[1:]):
    st,t,K,c=solve_t(greq)
    res=np.abs(Amap@K+Cmap@c-(t*Gc-Fc)).max()
    print(greq,st,"t^(1/4)=%.10f"%t**0.25,"eq.resid",res,flush=True)
    pickle.dump((t,K,c),open(f"sol_{greq:g}.pkl","wb"))
