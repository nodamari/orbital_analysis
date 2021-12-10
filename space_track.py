"""
This script requires tle_obj.py to run
"""
import datetime as dt
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from tle_obj import TLE
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


earth_radius = 6378.0 # km
earth_mu = 398600.0 # km^3 / s^2
J2 = 1.08262668e-3


def circle(r):
    """
    Creates a circle of radius r. Used for plotting the equator.
    :param
        r(float): radius of circle
    :return
        circle_arr(array): 3D array of circle of radius r
    """
    x = r * np.cos(np.linspace(-np.pi, 0, 700))
    x_return = x[len(x): 0: -1]

    y = np.sqrt(r**2 - np.multiply(x, x))
    y_return = -np.sqrt(r ** 2 - np.multiply(x_return, x_return))

    x = np.concatenate((x, x_return))
    y = np.concatenate((y, y_return))
    z = np.zeros(len(x))
    circle_arr = np.vstack((x, y, z))
    return circle_arr


def tle_cleanup(tle):
    """
    Takes the two line string TLE and converts it to a list containing the data from the TLE.
    inputs
        tle(str): Two-line elements created using the spacetrack package from space-track.org

    outputs
        tle(list): Two-line elements in a list

    Example:
    input
        1 25544U 98067A   21330.52589841  .00021613  00000-0  40355-3 0  9994
        2 25544  51.6433 260.0308 0004438 278.8265 247.0955 15.48696306313759
    output
        ['1', '25544U', '98067A', '21330.52589841', '.00021613', '00000-0', '40355-3', '0', '9994', '2', '25544', '51.6433', '260.0308', '0004438', '278.8265', '247.0955', '15.48696306313759']
    """
    tle = tle.split('\n')
    l1 = tle[0].split(' ')
    while '' in l1:
        l1.remove('')
    l2 = tle[1].split(' ')
    while '' in l2:
        l2.remove('')
    tle = l1 + l2
    return tle


def diffy_q(t, y, mu):
    # unpacking the elements in state
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])

    # norm of radius vector
    norm_r = np.linalg.norm(r)

    # two-body acceleration
    ax, ay, az = -r*mu/norm_r**3

    r_J2 = -(3 / 2) * J2 * (earth_mu / norm_r ** 2) * ((earth_radius / norm_r) ** 2)
    rx_J2 = r_J2 * (r[0] / norm_r) * ((5 * ((r[2] / norm_r) ** 2)) - 1)
    ry_J2 = r_J2 * (r[1] / norm_r) * ((5 * ((r[2] / norm_r) ** 2)) - 1)
    rz_J2 = r_J2 * (r[2] / norm_r) * ((5 * ((r[2] / norm_r) ** 2)) - 3)

    ax += rx_J2
    ay += ry_J2
    az += rz_J2

    return [vx, vy, vz, ax, ay, az]


if __name__ == '__main__':
    # get username and password from userpass.txt file
    with open('userpass.txt') as f:
        contents = f.readlines()
    user = contents[0].rstrip('\n')
    password = contents[1]

    st = SpaceTrackClient(user, password)
    drange = op.inclusive_range(dt.datetime(2021, 11, 26, 12),  dt.datetime(2021, 11, 27, 12))

    drange_next = op.inclusive_range(dt.datetime(2021, 11, 28, 12),  dt.datetime(2021, 11, 29, 12))
    iss_tle = st.tle(norad_cat_id=[25544], epoch=drange, limit=1,  format='tle')
    iss_next_tle = st.tle(norad_cat_id=[25544], epoch=drange_next, limit=1,  format='tle')
    iss_tle = tle_cleanup(iss_tle)
    iss_next_tle = tle_cleanup(iss_next_tle)


    """Space-Track data"""
    # create TLE object and create arrays for plotting
    iss = TLE(iss_tle)
    iss_path = iss.orbit
    current_pos = iss.pos_arr

    iss_next = TLE(iss_next_tle)
    iss_next_path = iss_next.orbit
    next_pos = iss_next.pos_arr

    equator = circle(earth_radius)


    """Numerical data"""
    # intial position and velocity vectors
    r0 = [iss.pos_arr[0][0], iss.pos_arr[1][0], iss.pos_arr[2][0]]
    v0 = [iss.vel_arr[0][0], iss.vel_arr[1][0], iss.vel_arr[2][0]]

    # time span
    tspan = 48*60*60

    # time step
    dt = 50

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


    """Plotting"""
    # initialize the figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plot orbit of object and equator
    plt.plot(iss_path[0], iss_path[1], iss_path[2], 'm-', label='ISS')
    ax.plot(iss_next_path[0], iss_next_path[1], iss_next_path[2], 'b-', label='ISS 48 hours')
    plt.plot(current_pos[0], current_pos[1], current_pos[2], 'mo', label='ISS current position')
    ax.plot(rs[:, 0], rs[:, 1], rs[:, 2], 'c', label='Numerically calculated trajectory')
    plt.plot(equator[0], equator[1], equator[2], 'r--', label='Equator')
    plt.legend()

    # plotting Earth
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    _x = earth_radius * np.cos(_u) * np.sin(_v)
    _y = earth_radius * np.sin(_u) * np.sin(_v)
    _z = earth_radius * np.cos(_v)

    # axes label
    ax.set_xlabel(['X (km)'])
    ax.set_ylabel(['Y (km)'])
    ax.set_zlabel(['Z (km)'])
    plt.legend()
    ax.plot_surface(_x,_y,_z, cmap='Blues', label='Earth', zorder=1, alpha=0.5)

    plt.show()
