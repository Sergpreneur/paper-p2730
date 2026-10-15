import numpy as np, mpmath as mp
mp.mp.dps=80
n=31
h=[mp.mpf(2)/n*mp.fsum(mp.sin(2*mp.pi*k*j/n) for k in range(1,16)) for j in range(n)]
Hm=mp.matrix(n,n)
for x in range(n):
    for y in range(n): Hm[x,y]=h[(x-y)%n]
f0=np.load("fstar.npy")
# symmetric reps: pairs {x,3-x}
reps=[];seen=set()
for x in range(n):
    y=(3-x)%n
    if x not in seen: reps.append((x,y)); seen|={x,y}
# reps[0]=(0,3) fixed c=1 ; fixed point pair (17,17) solved from zero mean
ip=[i for i,(x,y) in enumerate(reps) if x==y][0]
free=[i for i in range(len(reps)) if i not in (0,ip)]
A=mp.matrix(n,len(free)); b=mp.matrix(n,1)
for (x,y) in [reps[0]]: b[x]=1; b[y]=1
b[reps[ip][0]]=-2  # mean zero: 2*1 + ... + c17 = 0
for k,i in enumerate(free):
    x,y=reps[i]; A[x,k]+=1; A[y,k]+=1; A[reps[ip][0],k]-=2
def grad(*c):
    c=mp.matrix(list(c)); f=A*c+b
    g=Hm*f; F=mp.fsum(v**4 for v in g); G=mp.fsum(v**4 for v in f); R=F/G
    gf=Hm.T*mp.matrix([v**3 for v in g]) - R*mp.matrix([v**3 for v in f])
    return [x for x in (A.T*gf)]
c0=[mp.mpf(f0[reps[i][0]]) for i in free]
c=mp.findroot(grad,c0,tol=mp.mpf(10)**-70,maxsteps=50)
f=A*mp.matrix(c)+b
g=Hm*f; R=mp.fsum(v**4 for v in g)/mp.fsum(v**4 for v in f)
print("C^4 =",mp.nstr(R,60)); print("C   =",mp.nstr(R**0.25,60))
# full 30-dim stationarity residual and Hessian of R on zero-mean subspace (numerical, high precision)
gfull=Hm.T*mp.matrix([v**3 for v in g]) - R*mp.matrix([v**3 for v in f])
P=mp.eye(n)-mp.ones(n,n)/n
print("full projected gradient residual:",mp.nstr(mp.norm(P*gfull),5))
import pickle; pickle.dump(([str(v) for v in f],str(R)),open("fstar_mp.pkl","wb"))
