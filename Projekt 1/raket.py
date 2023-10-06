import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# m(t)a(t) = F + m'(t)u(t)
# F = m(t)g - c ||v(t)||v(t)

# t för tid
# v för hastighet
# c för luftmotståndskonstant
# k för hastighet för bränslepartiklar
def f(t, u, c, k, theta_m): #vår a(t)
    px, py, vx, vy = u

    m = 8 - 0.4 * t if t <= 10 else 4   # massa
    mprim = -0.4 if t <= 10 else 0      # takten med vilken massan ändras
    g = (0, -9.8)                       # graviationskonstanten

    g1, g2 = g
    
    #F = m * g - c * np.abs(v) * v
    F1, F2 = (m * g1 - c * np.abs(vx) * vx, m * g2 - c * np.abs(vy) * vy)
    
    #naiv lösning
    # theta = -(np.pi / 2) if pos[1] < 20 else np.arctan2( goal[1] - pos[1], goal[0] - pos[0] ) + np.pi

    theta = -(np.pi / 2) if py < 20 else theta_m

    u1, u2 = (k * np.cos(theta), k * np.sin(theta))

    return (vx, vy,(F1 + mprim * u1) / m, (F2 + mprim * u2) / m)

def runge_kutta(f, t, u_0, args):
    u = np.zeros((len(t),len(u_0)))
    u[0] = u_0
    for i in range(len(t)-1):
        K1 = f(t[i], u[i], *args)

        K2 = f(t[i+1], u[i] + np.multiply(K1, h), *args)

        u[i+1] = u[i] + (h/2) * (np.add(K1,K2))
    return u

def find_best_theta():
    min_theta = 0
    max_theta = 2 * np.pi

    #undersök olika värden med ökande precision
    for _ in range(4):
        step = (max_theta - min_theta) / 10
        thetas_to_examine = np.arange(min_theta,max_theta,step)
        min_distance = 1000

        for t in thetas_to_examine:
            #ingen max_step så funktionen är snabb men lite grov
            sol = scipy.integrate.solve_ivp(f,t_span, u_0, method="RK45", args=(c,k,t))

            positions = zip(sol.y[0], sol.y[1])
            goal_x, goal_y = goal
            distances = [np.sqrt((x - goal_x)**2 + (y - goal_y)**2) for x,y in positions]

            if min(distances) < min_distance:
                min_distance = min(distances)
                best_theta = t
        #nästa range att undersöka
        min_theta = best_theta - step 
        max_theta = best_theta + step

    print("best theta:",best_theta)
    return best_theta
            

goal = (80,60)
c = 0.05
k = 700
h = 1.5

t_0 = 0.0 # starttid
t_1 = 30.0 # sluttid
t_span = (t_0, t_1)
t_range = np.arange(t_0,t_1,h)
u_0 = [0,0,0,0] #x, y, vx, vy

theta_m = find_best_theta()

#hitta lösningar
rk_solution = runge_kutta(f,t_range,u_0, (c,k,theta_m))

ivp_solution = scipy.integrate.solve_ivp(f,t_span, u_0, method="RK45", args=(c,k,theta_m), max_step = h)

def animate_solution():
    #starta upp animation
    fig, axes = plt.subplots()
    axes.set(xlim=[-100, 700], ylim=[0, 800])

    #plotta rk_solution
    rk_tail = axes.plot((0,0) ,"r", linestyle = "-")
    rk_artist = axes.plot(0, 0, "b.", label="Vår Runge-Kutta")

    #plotta ivp_solution
    ivp_tail = axes.plot((0,0) ,"b", linestyle = "-")
    ivp_artist = axes.plot(0, 0, "g.", label="Inbyggd lösare")

    artists = [rk_artist, ivp_artist]
    tails = [rk_tail, ivp_tail]

    goal_artist = axes.plot(goal[0], goal[1], "rx", label="Mål")

    axes.legend()

    #update funktionen definierar hur animationen ser ut
    def update(frame):
        #vår lösares positioner och solve_ivps positioner har olika "shape"
        rk_position = (rk_solution[frame][0], rk_solution[frame][1])
        ivp_position = (ivp_solution.y[0][frame], ivp_solution.y[1][frame])

        positions = [rk_position, ivp_position]
        
        #animerar båda raketerna
        for artist, tail, pos  in zip(artists, tails, positions):
            px, py = pos

            artist[0].set_data((px, py))

            theta = -(np.pi / 2) if py < 20 else theta_m

            rx, ry = (px + 10 *np.cos(theta), py + 10 * np.sin(theta))
            tail[0].set_data([(px,rx),(py,ry)])

    ani = animation.FuncAnimation(fig = fig, func = update, interval = 1, frames = range(len(t_range)))
    ani.save(f"{goal} h={h}.gif", writer="ImageMagickWriter")
    # plt.show()

def plot_solution():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    fig.set_size_inches(9,9)
    fig.suptitle(f'Goal vid {goal}, h={h}', fontsize=16)

    ax1.set_title("Inbyggd lösare position")
    ax2.set_title("Inbyggd lösare hastighet")
    
    ax1.plot(ivp_solution.t, ivp_solution.y[0], label="x")
    ax1.plot(ivp_solution.t, ivp_solution.y[1], label="y")
    ax1.legend()
    
    ax2.plot(ivp_solution.t, ivp_solution.y[2], label="x velocity")
    ax2.plot(ivp_solution.t, ivp_solution.y[3], label="y velocity")
    ax2.legend()

    ax3.set_title("Vår lösare position")
    ax4.set_title("Vår lösare hastighet")

    rk_plot = np.transpose(rk_solution)
    ax3.plot(t_range, rk_plot[0], label="x")
    ax3.plot(t_range, rk_plot[1], label="y")
    ax3.legend()
    ax4.plot(t_range, rk_plot[2], label="x velocity")
    ax4.plot(t_range, rk_plot[3], label="y velocity")
    ax4.legend()
    plt.show()

animate_solution()