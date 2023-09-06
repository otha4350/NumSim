import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

pos = (0,0)
goal = (70,60)

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
def f(t, u, c, k, theta_m): #vår a(t)
    px, py, vx, vy = u

    m = 8 - 0.4 * t if t <= 10 else 4   # massa
    mprim = -0.4 if t <= 10 else 0      # takten med vilken massan ändras
    g = (0, -9.8)                       # graviationskonstanten

    g1, g2 = g
    
    #F = m * g - c * numpy.abs(v) * v
    F1, F2 = (m * g1 - c * numpy.abs(vx) * vx, m * g2 - c * numpy.abs(vy) * vy)
    
    #naiv lösning
    # theta = -(numpy.pi / 2) if pos[1] < 20 else numpy.arctan2( goal[1] - pos[1], goal[0] - pos[0] ) + numpy.pi

    theta = -(numpy.pi / 2) if py < 20 else theta_m

    u1, u2 = (k * numpy.cos(theta), k * numpy.sin(theta))

    return (vx, vy,(F1 + mprim * u1) / m, (F2 + mprim * u2) / m)

t0 = 0.0 # starttid
t1 = 45.0 # sluttid
tspan = (t0, t1)
h = 0.1
t_range = numpy.arange(t0,t1,h)
u_0 = [0,0,0,0]


u = [u_0]
for i, t in enumerate(t_range[:-1]):
    K1 = f(t, u[i], c , k, 110)

    K2 = f(t_range[i+1], numpy.add(u[i], numpy.multiply(K1, h)) , c , k, 110)

    u.append(numpy.add(u[i], (h/2) * (numpy.add(K1,K2))))

best_theta = 0
best_min_goal_dist = 1000
for deg in numpy.arange(343.5,344.5,0.01):
    print(deg)
    theta_m = deg

    sol = scipy.integrate.solve_ivp(f,tspan, u_0, method="RK45", args=(c,k,theta_m), max_step = h)
    pxs = sol.y[0]
    pys = sol.y[1]

    min_goal_dist = 1000

    px, py = pxs[-1], pys[-1]
    goal_dist = numpy.sqrt((px - goal[0])**2 + (py - goal[1])**2)
    
    if goal_dist < min_goal_dist:
        min_goal_dist = goal_dist
    
    if min_goal_dist < best_min_goal_dist:
        best_min_goal_dist = min_goal_dist
        best_theta = theta_m

print("best theta:", best_theta)

sol = scipy.integrate.solve_ivp(f,tspan, u_0, method="RK45", args=(c,k,best_theta), max_step = h)
    



# fig, ((axes, ax1), (ax2, ax3)) = plt.subplots(2,2)
fig, axes = plt.subplots()
aim = axes.plot((0,0) ,"r", linestyle = "-")
accurate_aim = axes.plot((0,0) ,"b", linestyle = "-")

scat = axes.plot(0, 0, "b.")
accurate = axes.plot(0, 0, "g.")

rockets = [scat, accurate]
tails = [aim, accurate_aim]

goal = axes.plot(goal[0], goal[1], "rx")



axes.set(xlim=[-100, 100], ylim=[-100, 800])
def update(frame):
    positions = [(u[frame][0], u[frame][1]), (sol.y[0][frame], sol.y[1][frame])]

    
    for rocket, tail, pos  in zip(rockets, tails, positions):
        px, py = pos

        rocket[0].set_data(px, py)

        theta = -(numpy.pi / 2) if py < 20 else numpy.arctan2( 60 - py, 80 - px ) + numpy.pi

        rx, ry = (px + 10 *numpy.cos(theta), py + 5 * numpy.sin(theta))
        tail[0].set_data([(px,rx),(py,ry)])

    return scat

ani = animation.FuncAnimation(fig = fig, func = update, interval = 1, repeat = True)

# axes.plot(sol.y[1])
# ax2.plot(rk_positions)
plt.show()