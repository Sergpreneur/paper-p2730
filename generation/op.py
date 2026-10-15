import numpy as np
n=31
h=np.zeros(n)
for j in range(n):
    h[j]=2/n*sum(np.sin(2*np.pi*k*j/n) for k in range(1,16))
h3=np.array([0]+[(np.cos(np.pi*j/n)-(-1)**j)/(n*np.sin(np.pi*j/n)) for j in range(1,n)])
H=np.array([[h[(x-y)%n] for y in range(n)] for x in range(n)])
