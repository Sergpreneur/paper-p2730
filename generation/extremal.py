import numpy as np, mpmath as mp
from scipy.optimize import minimize
from op import H, n
mp.mp.dps=60
P0=np.ones((n,n))/n; B0=np.linalg.svd(np.eye(n)-P0)[0][:,:30]
def neg(x): f=B0@x; return -np.sum((H@f)**4)/np.sum(f**4)
rng=np.random.default_rng(5); sols=[]
for _ in range(400):
    r=minimize(neg,rng.standard_normal(30),method="BFGS",options={"gtol":1e-12})
    sols.append((-r.fun,B0@r.x))
vals=np.array([s[0] for s in sols])**0.25
u,cnt=np.unique(np.round(vals,9),return_counts=True)
print("local maxima (value: hits):")
for a,b in sorted(zip(u,cnt),reverse=True)[:8]: print(f"  {a:.9f}: {b}")
f=max(sols,key=lambda s:s[0])[1]; f=f/np.abs(f).max()
# normalize orbit: shift so argmax at 0, check reflection symmetry f(-x)=±f(x)
k=int(np.argmax(np.abs(f))); f=np.roll(f,-k)*np.sign(f[k])
print("f:",np.round(f,6))
print("even part norm",np.linalg.norm((f+np.roll(f[::-1],1))/2),"odd part norm",np.linalg.norm((f-np.roll(f[::-1],1))/2))
# best reflection axis
best=min(((np.linalg.norm(f-s*np.roll(f[::-1],a)),a,s) for a in range(n) for s in (1,-1)))
print("closest reflection symmetry f(x)=s f(a-x):",best)
np.save("fstar.npy",f)
