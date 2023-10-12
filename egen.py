import numpy as np
import random
import matplotlib.pyplot as plt
import math
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


beta = 0.3
gamma = 1/14
alpha = 1/4
mu = 1/40
v = 1
misinformation = 0.01
v1Effectiveness = 0.5
v2Effectiveness = 0.9
vaccineIncubationTime = 1/2

n = 1000
# [antivaxx, susceptible, exposed, infected, vaccinated1, v1exposed, v1infected, vaccinated2, v2exposed, v2infected, recovered, deceased, vaxximmune]
y_0 =[5, n-10, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]

t_0 = 0.0 # starttid
t_1 = 120.0 # sluttid
t_span = (t_0, t_1)
h = 0.01

#översätt till stökiomatri och propensitet
# sp = - beta * i * s * (1/n)
# ip = beta * i * s * (1/n) - gamma * i
# rp = gamma * i

def stoch():
# [antivax, sus, exposed, infected, vaccinated1, v1exposed, v1infected, vaccinated2, v2exposed, v2infected, recovered, deceased, vaxximmune]
    return np.array([
        [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #sus->exposed
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #exposed->infected
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0], #infected->recovered
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0], #infectius -> deceased
        [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #sus -> vaccinated1
        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0], #vaccinated1 -> v1exposed
        [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0], #v1exposed -> v1infected
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0], #v1infected -> recovered
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0], #v1infected -> deceased
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1], #vaccinated1 -> vaxximmune
        [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0], #vaccinated1 -> vaccinated2
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0], #vaccinated2 -> v2exposed
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0], #v2exposed -> v2infected
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0], #v2infected -> recovered
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0], #v2infected -> deceased
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], #vaccinated2 -> vaxximmune
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #sus->antivax
        [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #antivax->exposed
    ])

coeff = np.array([beta,alpha, gamma, mu, v, misinformation,v1Effectiveness,v2Effectiveness,vaccineIncubationTime])

def prop(X, coeff):
    n = sum(X)
    antivaxx, susceptible, exposed, infected, vaccinated1, v1exposed, v1infected, vaccinated2, v2exposed, v2infected, recovered, deceased, vaxximmune = X
    beta, alpha, gamma, mu, v, misinformation, v1Effectiveness,v2Effectiveness,vaccineIncubationTime = coeff

    totalinfected = sum([infected,v1infected,v2infected])
    w = np.array([
        beta * (totalinfected/n) * susceptible, #sus->exposed
        alpha * exposed, #exposed->infected
        gamma * infected, #infected->recovered
        mu * infected, #infectius -> deceased
        v if susceptible > 0 else 0, #sus -> vaccinated1
        (1 - v1Effectiveness) * beta * (totalinfected/n) * vaccinated1, #vaccinated1 -> v1exposed
        alpha * v1exposed, #v1exposed -> v1infected
        gamma * v1infected, #v1infected -> recovered
        (1 - v1Effectiveness) * mu * v1infected, #v1infected -> deceased
        v1Effectiveness * vaccinated1, #vaccinated1 -> vaxximmune
        vaccineIncubationTime * vaccinated1, #vaccinated1 -> vaccinated2
        (1 - v2Effectiveness) * beta * (totalinfected/n) * vaccinated2, #vaccinated2 -> v2exposed
        alpha * v2exposed, #v2exposed -> v2infected
        gamma * v2infected, #v2infected -> recovered
        (1-v2Effectiveness) * mu * infected, #v2infected -> deceased
        v2Effectiveness * vaccinated2, #vaccinated2 -> vaxximmune
        misinformation * susceptible, #sus->antivax
        beta * (totalinfected/n) * antivaxx, #antivax->exposed
    ])

    return w

tvec, sols = SSA(prop, stoch, y_0, t_span, coeff)

sols = np.transpose(sols)

fig, axes = plt.subplots()

axes.plot(tvec, sols[0],label="Antivax")
axes.plot(tvec, sols[1],label="Susceptible")
axes.plot(tvec, sols[2]+sols[5]+sols[8],label="Exposed")
axes.plot(tvec, sols[3]+sols[6]+sols[9],label="Infected", color="red")
axes.plot(tvec, sols[4],label="Vaccinated1")
# axes.plot(tvec, sols[5],label="V1Exposed")
# axes.plot(tvec, sols[6],label="V1infected")
axes.plot(tvec, sols[7],label="Vaccinated2")
# axes.plot(tvec, sols[8],label="V2Exposed")
# axes.plot(tvec, sols[9],label="V2infected")
axes.plot(tvec, sols[10],label="Recovered", color="blue")
axes.plot(tvec, sols[11],label="Deceased", color="black")
axes.plot(tvec, sols[12],label="Immunized", color="cyan")

#skriv ut maximala antalet samtidigt infekterade
print(max(sols[3]+sols[6]+sols[9]))

plt.legend()
plt.show()
