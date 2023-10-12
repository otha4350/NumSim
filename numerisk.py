import numpy as np
import scipy
import matplotlib.pyplot as plt

def f(t, y, beta, gamma):
    s, i, r = y
    n = sum(y)

    sp = - beta * i * s * (1/n)
    ip = beta * i * s * (1/n) - gamma * i
    rp = gamma * i

    return (sp, ip, rp)

beta = .3
gamma = 1/7
y0 = (995, 5, 0)

t_0 = 0.0 # starttid
t_1 = 120.0 # sluttid
t_span = (t_0, t_1)
h = 0.1

ivp_solution = scipy.integrate.solve_ivp(f,t_span, y0, method="RK45", args=(beta, gamma), max_step=h)

fig, axes = plt.subplots()

axes.plot(ivp_solution.t, ivp_solution.y[0], label="S")
axes.plot(ivp_solution.t, ivp_solution.y[1], label="I")
axes.plot(ivp_solution.t, ivp_solution.y[2], label="R")
plt.legend()
plt.show()