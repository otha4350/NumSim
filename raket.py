import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

pos = (0,0)
goal = (80,60)

c = 0.05
k = 700

# m(t)a(t) = F + m'(t)u(t)
# F = m(t)g - c ||v(t)||v(t)

# t för tid
# v för hastighet
# c för luftmotståndskonstant
# k för hastighet för bränslepartiklar
positions = []
prev_t = 0
def f(t, v, c, k): #vår a(t)
    m = 8 - 0.4 * t if t <= 10 else 4   # massa
    mprim = -0.4 if t <= 10 else 0      # takten med vilken massan ändras
    g = (0, -9.8)                       # graviationskonstanten
    
    global pos
    global prev_t
    global positions
    pos = (pos[0] + v[0] * (t - prev_t), pos[1] + v[1] * (t - prev_t))
    prev_t = t
    print(f'({round(pos[0])},{round(pos[1])})')
    positions.append(pos)

    v1, v2 = v
    g1, g2 = g
    
    #F = m * g - c * numpy.abs(v) * v
    F1, F2 = (m * g1 - c * numpy.abs(v1) * v1, m * g2 - c * numpy.abs(v2) * v2)
    
    #naiv lösning
    # theta = -(numpy.pi / 2) if pos[1] < 20 else numpy.arctan2( goal[1] - pos[1], goal[0] - pos[0] ) + numpy.pi

    theta = -(numpy.pi / 2) if pos[1] < 20 else numpy.arctan2( goal[1] - pos[1], goal[0] - pos[0] ) + numpy.pi

    u1, u2 = (k * numpy.cos(theta), k * numpy.sin(theta))

    return ((F1 + mprim * u1) / m, (F2 + mprim * u2) / m)

t0 = 0.0 # starttid
t1 = 30.0 # sluttid
tspan = (t0, t1)
h = 0.01
t_range = numpy.arange(t0,t1,h)
v_0 = (0,0)

y = [v_0]
for i, t in enumerate(t_range[:-1]):
    K1 = f(t, y[i], c , k)
    K11, K12 = K1

    K2 = f(t_range[i+1], (y[i][0] + h * K11, y[i][1] + h * K12), c , k)
    K21, K22 = K2

    y.append((y[i][0] + (h/2) * (K11 + K21), y[i][1] + (h/2) * (K12 + K22)))

rk_positions = [(0,0)]
for dx, dy in y:
    rk_positions.append((rk_positions[-1][0] + dx * h,rk_positions[-1][1] + dy * h))
    

# sol = scipy.integrate.solve_ivp(f,tspan, v_0, method="RK45", args=(c,k), max_step = 0.05)
# print(sol)
# print(sol.y)


fig, axes = plt.subplots()
scat = axes.plot(0, 0, "bo")
goal = axes.plot(goal[0], goal[1], "ro")

axes.set(xlim=[-100, 100], ylim=[-5, 100])
def update(frame):
    scat[0].set_data(positions[frame * 5])

    # scat[0].set_data(rk_positions[frame * 5])

    return scat

ani = animation.FuncAnimation(fig = fig, func = update, interval = 1)

# axes.plot(t_range,y)
# axes.plot(rk_positions)
plt.show()