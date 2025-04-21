"""
This script requires tle_obj.py and  userpass.txt files to run.
tle_obj.py is a class object file.
userpass.txt is a text file containing the username for space-track.org in the first line and the password in the
second line.
"""
import datetime as dt
import math
import pickle

from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from tle_obj import TLE
from orbital_elements import OE
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d.axes3d import Axes3D
#plt.style.use('dark_background')
# plt.rcParams.update({
#     "lines.color": "dimgray",
#     "patch.edgecolor": "black",
#     "text.color": "lightgray",
#     "axes.facecolor": "black",
#     "axes.edgecolor": "dimgray",
#     "axes.labelcolor": "darkgray",
#     "xtick.color": "darkgray",
#     "ytick.color": "darkgray",
#     "grid.color": "gray",
#     "figure.facecolor": "black",
#     "figure.edgecolor": "black",
#     "savefig.facecolor": "black",
#     "savefig.edgecolor": "black"})


earth_radius = 6378.0 # km
earth_mu = 398600.4418 # km^3 / s^2
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


def air_density_ratio(r):
    R = 8.314
    g = 9.81
    M = 0.029
    B = 0.0065
    T_o = 288.16

    H = (R*(T_o - (B*r)))/(M*g)

    rho_o = 0.01569659023
    alt = earth_radius - r

    ratio = math.exp(alt/H)
    return ratio

def angle(v1, v2):
    v1 = np.ndarray.flatten(v1)
    v2 = np.ndarray.flatten(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)
    # unit_v1 = v1 / np.linalg.norm(v1)
    # unit_v2 = v2 / np.linalg.norm(v2)
    # dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg



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


    if len(tle) == 0:
        raise ValueError('Invalid query. Check if the satellite ID is correct and/or try a different date time range.')

    tle = tle.split()

    # if some of the elements are concatenated
    # usually the checksum number is combined with the last element in each line
    if len(tle) < 20:
      second_line_idx = tle.index('2') # find the list index of the '2' line num
      element_checksum_1_idx = second_line_idx -1
      element_num = tle[element_checksum_1_idx][0:-1]
      checksum_1 = tle[element_checksum_1_idx][-1]

      rev_epoch = tle[-1][0:-1]
      checksum_2 = tle[-1][-1]

      tle[element_checksum_1_idx: element_checksum_1_idx+1] = element_num, checksum_1
      tle[len(tle)-1 : len(tle)] = rev_epoch, checksum_2

    return tle

def fun(S, t, mu):
    rx, ry, rz, vx, vy, vz = S

    r = np.array([rx, ry, rz])

    # norm of radius vector
    norm_r = np.linalg.norm(r)

    # Earth's gravitational acceleration
    ax, ay, az = -(r * mu) / (norm_r ** 3)

    return [vx, vy, vz, ax, ay, az]


def diffy_q(t, y, mu, bstar):
    # unpacking the elements in state
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])
    # norm of radius vector
    norm_r = np.linalg.norm(r)

    # Earth's gravitational acceleration
    ax, ay, az = -(r*mu)/(norm_r**3)

    # J2 correction
    r_J2 = -(3 / 2) * J2 * (earth_mu / norm_r ** 2) * ((earth_radius / norm_r) ** 2)
    rx_J2 = r_J2 * (r[0] / norm_r) * ((5 * ((r[2] / norm_r) ** 2)) - 1) * 0.0001
    ry_J2 = r_J2 * (r[1] / norm_r) * ((5 * ((r[2] / norm_r) ** 2)) - 1) * 0.0001
    rz_J2 = r_J2 * (r[2] / norm_r) * ((5 * ((r[2] / norm_r) ** 2)) - 3) * 0.0001


    # atmospheric drag
    drag_x = air_density_ratio(norm_r) * bstar / earth_radius * (vx ** 2)
    drag_y = air_density_ratio(norm_r) * bstar / earth_radius * (vy ** 2)
    drag_z = air_density_ratio(norm_r) * bstar / earth_radius * (vz ** 2)

    #ax += rx_J2 #+ drag_x
    #ay += ry_J2 #+ drag_y
    #az += rz_J2 #+ drag_z

    # ax += drag_x
    # ay += drag_y
    # az += drag_z
    return [vx, vy, vz, ax, ay, az]


if __name__ == '__main__':
    #get username and password from userpass.txt file
    with open('userpass.txt') as f:
        contents = f.readlines()
    user = contents[0].rstrip('\n')
    password = contents[1]

    st = SpaceTrackClient(user, password)
    sat_cat_id = 44952# 25544 # ISS

    drange = op.inclusive_range(dt.datetime(2025, 4, 12),  dt.datetime(2025, 4, 13))
    #drange_next = op.inclusive_range(dt.datetime(2021, 10, 14), dt.datetime(2021, 10, 15))

    drange_next = op.inclusive_range(dt.datetime(2025, 4, 14),  dt.datetime(2025, 4, 15))
    sat_tle = st.tle(norad_cat_id=[sat_cat_id], epoch=drange, limit=1,  format='tle')
    sat_next_tle = st.tle(norad_cat_id=[sat_cat_id], epoch=drange_next, limit=1,  format='tle')

    sat_tle =  tle_cleanup(sat_tle)
    sat_next_tle = tle_cleanup(sat_next_tle)
    print(sat_tle)
    print(sat_next_tle)


    # with open("sat.pkl", "rb") as inp:
    #     sat = pickle.load(inp)
    #     sat_next = pickle.load(inp)
    """Space-Track data"""
    # create TLE object and create arrays for plotting
    sat = TLE(sat_tle)
    sat_path = sat.orbit
    current_pos = sat.pos_arr

    sat_next = TLE(sat_next_tle)
    sat_next_path = sat_next.orbit
    next_pos = sat_next.pos_arr

    equator = circle(earth_radius)

    # with open("sat.pkl", "wb") as outf:
    #     pickle.dump(sat, outf)
    #     pickle.dump(sat_next, outf)


    """Numerical data"""
    # intial position and velocity vectors
    r0 = [sat.pos_arr[0][0], sat.pos_arr[1][0], sat.pos_arr[2][0]]
    v0 = [sat.vel_arr[0][0], sat.vel_arr[1][0], sat.vel_arr[2][0]]

    # time span
    d_epoch = sat_next.epoch - sat.epoch
    tspan = int(d_epoch)*24*60*60

    # time step
    dt = 50 # [sec]

    # total number of steps
    n_steps = int(np.ceil(tspan/dt))+1


    # intialize arrays
    ys = np.zeros((n_steps, 6))
    ts = np.zeros((n_steps, 1))

    # intiial contidions
    y0 = r0 + v0 # [rx, ry, rz, vx, vy, vz]
    ys[0] = np.array(y0)  # sets initial values according to r0 and v0
    step = 1 # second step because first step defined as initial contidions

    # intialize solver
    solver = ode(diffy_q)
    solver.set_integrator('dopri5')
    solver.set_initial_value(y0, 0)
    solver.set_f_params(earth_mu, sat.bstar)

    # propagate orbit
    while solver.successful() and step < n_steps:
        solver.integrate(solver.t+dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step +=1
    rs = ys[:, :3]
    vs = ys[:, 3:]


    orbital_elements_converted = OE(rs, vs)
    print()

    plt.plot(ts, np.sqrt((rs[:, 0]**2) + (rs[:, 1]**2) + (rs[:, 2]**2)))
    plt.plot(ts, np.sqrt((vs[:, 0]**2) + (vs[:, 1]**2) + (vs[:, 2]**2)) *1000)
    plt.show()
    """
    # PLOTTING #
    
    # initialize the figure
    fig = plt.figure()
    ax = Axes3D(fig, box_aspect=(1, 1, .85))

    # plot orbit of object and equator
    plt.plot(sat_path[0], sat_path[1], sat_path[2], color='deepskyblue', linestyle='solid', linewidth=2,  label='sat')
    ax.plot(sat_next_path[0], sat_next_path[1], sat_next_path[2], color='hotpink', linestyle='solid', linewidth=2, label='sat {0:.2f} hours'.format(d_epoch*24))
    plt.plot(current_pos[0], current_pos[1], current_pos[2], marker='o', markerfacecolor='deepskyblue', markeredgecolor='deepskyblue', label='sat current position')
    ax.plot(rs[:, 0], rs[:, 1], rs[:, 2], color='blueviolet', linestyle='solid', alpha=0.5, label='Numerically calculated trajectory')
    #ax.plot(rs[1650:, 0], rs[1650:, 1], rs[1650:, 2], color='blueviolet', linestyle='solid', alpha=0.5, label='Numerically calculated trajectory')
    plt.plot(equator[0], equator[1], equator[2], color='limegreen', linestyle='dashed', label='Equator')
    ax.quiver(sat.pos_arr[0], sat.pos_arr[1], sat.pos_arr[2],  sat.vel_arr[0]*300,  sat.vel_arr[1]*300, sat.vel_arr[2]*300, color='deepskyblue')
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
    ax.plot_surface(_x,_y,_z, color='honeydew', label='Earth', zorder=1, alpha=0.3)

    plt.show()
    """