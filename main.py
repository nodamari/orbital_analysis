import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

earth_radius = 6378.0 # km
earth_mu = 398600.0 # km^3 / s^2



def diffy_q(t, y, mu):
    # unpacking the elements in state
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])

    # norm of radius vector
    norm_r = np.linalg.norm(r)

    # two-body acceleration
    ax, ay, az = -r*mu/norm_r**3

    return [vx,vy, vz, ax, ay, az]


def plot(rs):
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(rs[:,0], rs[:,1], rs[:,2], 'm', label='Trajectory')
    ax.plot([rs[0, 0]], [rs[0,1]], [rs[0, 2]], 'mo', label='Initial Position')

    # plot central body
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    _x = earth_radius * np.cos(_u) * np.sin(_v)
    _y = earth_radius * np.sin(_u) * np.sin(_v)
    _z = earth_radius * np.cos(_v)
    ax.plot_surface(_x,_y,_z, cmap='Blues')

    # plot the x, y, z, vectors
    l = earth_radius*2
    x, y, z = [[0, 0, 0], [0,0,0], [0, 0, 0]]
    u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]

    ax.quiver(x, y, z, u, v, w, colors='k')
    max_val = np.max(np.abs(rs))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    ax.set_xlabel(['X (km)'])
    ax.set_ylabel(['Y (km)'])
    ax.set_zlabel(['Z (km)'])


    #ax.set_aspect('equal')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    # initial conditions of orbit parameters
    r_mag = earth_radius + 500.0
    v_mag = np.sqrt(earth_mu/r_mag)

    # intial position and velocity vectors
    r0 = [r_mag, 0, 0]
    v0 = [0, v_mag, 0]

    # time span
    tspan = 100*60

    # time step
    dt = 100

    # total number of steps
    n_steps = int(np.ceil(tspan/dt))

    # intialize arrays
    ys = np.zeros((n_steps, 6))
    ts = np.zeros((n_steps, 1))

    # intiial contidions
    y0 = r0 + v0 # [rx, ry, rz, vx, vy, vz]
    ys[0] = np.array(y0)  # sets initial values according to r0 and v0
    step = 1 # second step because first step defined as initial contidions

    # intialize solver
    solver = ode(diffy_q)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0, 0)
    solver.set_f_params(earth_mu)

    # propagate orbit
    while solver.successful() and step < n_steps:
        solver.integrate(solver.t+dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step +=1
    rs = ys[:, :3]
    vs = ys[:, 3:]
    h = np.cross(rs, vs)
    print(h)
    plot(rs)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
