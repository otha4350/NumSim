#==================================
# Gillespies algoritm
# Numeriska metoder och Simulering
#==================================
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from time import sleep
def SSA(prop,stoch,X0,tspan,coeff):
    tvec=np.zeros(1)
    tvec[0]=tspan[0]
    Xarr=np.zeros([1,len(X0)])
    Xarr[0,:]=X0
    t=tvec[0]
    X=X0
    sMat=stoch()
    while t<tspan[1]:
        re=prop(X,coeff)
        a0=sum(re)
        if a0>1e-10:
            r1=random.random()
            r2=random.random()
            tau=-math.log(r1)/a0
            cre=np.cumsum(re)
            cre=cre/cre[-1]
            r=0
            while cre[r]<r2:
                r+=1
            t+=tau
            tvec=np.append(tvec,t)
            X=X+sMat[r,:]
            Xarr=np.vstack([Xarr,X])
        else:
            print('Simulation ended at t=', t)
            return tvec, Xarr
            
            
    return tvec, Xarr


beta = .3
gamma = 1/7
alpha = 1/9
mu = 1/20
n = 1000
y_0 = [n - 5, 0, 5, 0, 0]

t_0 = 0.0 # starttid
t_1 = 120.0 # sluttid
t_span = (t_0, t_1)
h = 0.1

#översätt till stökiomatri och propensitet
# sp = - beta * i * s * (1/n)
# ip = beta * i * s * (1/n) - gamma * i
# rp = gamma * i

def stoch():
    return np.array([
        [-1, 1, 0, 0, 0],    
        [0, -1, 1, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, -1, 0, 1]    
    ])

coeff = np.array([beta,alpha, gamma, mu])

def prop(X, coeff):
    n = sum(X)
    s, e, i, r, d = X
    beta, alpha, gamma, mu = coeff
    
    w = np.array([beta*(i/n)*s, alpha * e, gamma*i, mu*i])

    return w

tvec, sols = SSA(prop, stoch, y_0, t_span, coeff)

sols = np.transpose(sols)

fig, axes = plt.subplots()
axes.plot(tvec, sols[0],label="S")
axes.plot(tvec, sols[1],label="E")
axes.plot(tvec, sols[2],label="I")
axes.plot(tvec, sols[3],label="R")
axes.plot(tvec, sols[4],label="D")

#ju större dödlighet desto fler dör
print(max(sols[2]))

plt.legend()
plt.show()
