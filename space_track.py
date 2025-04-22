"""
This script requires tle_obj.py and  userpass.txt files to run.
tle_obj.py is a class object file.
userpass.txt is a text file containing the username for space-track.org in the first line and the password in the
second line.
"""
import datetime
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
    ax_J2 = r_J2 * (r[0] / norm_r) * ((-5 * ((r[2] / norm_r) ** 2)) + 1)
    ay_J2 = r_J2 * (r[1] / norm_r) * ((-5 * ((r[2] / norm_r) ** 2)) + 1)
    az_J2 = r_J2 * (r[2] / norm_r) * ((-5 * ((r[2] / norm_r) ** 2)) + 3)


    j2_comp = (3/2) * J2 * ((earth_radius/norm_r) **2)
    x_comp = - (earth_mu * r[0]) / (norm_r**3)
    y_comp = - (earth_mu * r[1]) / (norm_r ** 3)
    z_comp = - (earth_mu * r[2]) / (norm_r ** 3)
    xy = 1 - 5* ((r[2]**2)/(norm_r**2))
    z = 3 - 5* ((r[2]**2)/(norm_r**2))

    ax_J2 = x_comp * (1 + (j2_comp * xy))
    ay_J2 = y_comp * (1 + (j2_comp * xy))
    az_J2 = z_comp * (1 + (j2_comp * z))


    # atmospheric drag
    drag_x = air_density_ratio(norm_r) * bstar / earth_radius * (vx ** 2)
    drag_y = air_density_ratio(norm_r) * bstar / earth_radius * (vy ** 2)
    drag_z = air_density_ratio(norm_r) * bstar / earth_radius * (vz ** 2)

    #ax += ax_J2 #+ drag_x
    #ay += ay_J2 #+ drag_y
    #az += az_J2 #+ drag_z

    # ax += drag_x
    # ay += drag_y
    # az += drag_z
    return [vx, vy, vz, ax, ay, az]

def read_multiple_tles(tles):
    tle_list = []
    line_broken = tles.split("\n")
    for i in range(len(line_broken)):
        if i % 2 ==1:

            first_line = line_broken[i-1]
            second_line = line_broken[i]

            first_line = first_line.split()
            if len(first_line) < 10:
              first_checksum = first_line[-1][-1]
              element_num = first_line[-1][0:-1]
              first_line[len(first_line)-1: len(first_line)] = element_num, first_checksum

            second_line = second_line.split()

            if len(second_line) < 9:
              rev_epoch = second_line[-1][0:-1]
              checksum_2 = second_line[-1][-1]
              second_line[len(second_line) - 1: len(second_line)] = rev_epoch, checksum_2

            first_line.extend(second_line)
            tle_list.append(first_line)

    return tle_list


if __name__ == '__main__':
    #get username and password from userpass.txt file
    with open('userpass.txt') as f:
        contents = f.readlines()
    user = contents[0].rstrip('\n')
    password = contents[1]

    st = SpaceTrackClient(user, password)
    sat_cat_id = 44952# 25544 # ISS
    sat_cat_id = 44952# 25544 # ISS

    drange = op.inclusive_range(datetime.datetime(2025, 4, 15),  datetime.datetime(2025, 4, 16))
    #drange_next = op.inclusive_range(dt.datetime(2021, 10, 14), dt.datetime(2021, 10, 15))

    drange_next = op.inclusive_range(datetime.datetime(2025, 4, 17),  datetime.datetime(2025, 4, 18))
    sat_tle = st.tle(norad_cat_id=[sat_cat_id], epoch=drange, limit=1,  format='tle')
    sat_next_tle = st.tle(norad_cat_id=[sat_cat_id], epoch=drange_next, limit=1,  format='tle')

    sat_tle =  tle_cleanup(sat_tle)
    sat_next_tle = tle_cleanup(sat_next_tle)
    print(sat_tle)



    """Space-Track data"""
    # create TLE object and create arrays for plotting
    sat = TLE(sat_tle)
    sat_path = sat.orbit
    current_pos = sat.pos_arr

    sat_next = TLE(sat_next_tle)
    sat_next_path = sat_next.orbit
    next_pos = sat_next.pos_arr

    # equator = circle(earth_radius)

    # TLEs for verification
    two_day  = op.inclusive_range(datetime.datetime(2025, 4, 15),  datetime.datetime(2025, 4, 17, 15))
    two_day_tles = st.tle(epoch=two_day, format="tle", limit=12, norad_cat_id=sat_cat_id)
    tle_list = read_multiple_tles(two_day_tles)
    tle_dict = {}

    first_epoch = -1
    for tle in tle_list:
        tle_object = TLE(tle)
        tle_dict[tle_object.epoch] = {}
        tle_dict[tle_object.epoch]["argument_perigee"] = tle_object.arg_perigee
        tle_dict[tle_object.epoch]["eccentricity"] = tle_object.eccentricity
        tle_dict[tle_object.epoch]["true_anomaly"] = tle_object.true_anomaly()
        tle_dict[tle_object.epoch]["raan"] = tle_object.right_ascension
        tle_dict[tle_object.epoch]["inclination"] = tle_object.inclination
        tle_dict[tle_object.epoch]["pos_vec"] = tle_object.pos_arr
        tle_dict[tle_object.epoch]["vel_vec"] = tle_object.vel_arr
        tle_dict[tle_object.epoch]["distance"] = np.linalg.norm(tle_object.pos_arr)
        tle_dict[tle_object.epoch]["speed"] = np.linalg.norm(tle_object.vel_arr)
        tle_dict[tle_object.epoch]["h_vec"] = tle_object.h

        year = int("20" + str(tle_object.epoch)[0:2])
        days_into_year = tle_object.epoch % 1000
        tle_dict[tle_object.epoch]["datetime_obj"] = datetime.datetime(year=year, month=1, day=1) + datetime.timedelta(days=days_into_year-1)

        if len(tle_dict) == 1:
            first_epoch = tle_dict[tle_object.epoch]["datetime_obj"]
            tle_dict[tle_object.epoch]["relative_time"] = 0
        else:
            relative_epoch = tle_dict[tle_object.epoch]["datetime_obj"] - first_epoch
            tle_dict[tle_object.epoch]["relative_time"] = (relative_epoch.days * 24) + ((relative_epoch.seconds + (relative_epoch.microseconds/1000000)) / 3600) # [hr]


    relative_time_list = []
    distance_list = []
    speed_list = []
    true_anomaly_list = []
    raan_list = []
    arg_perigee_list = []
    inclination_list = []
    for epoch, values in tle_dict.items():
        relative_time_list.append(values["relative_time"])
        distance_list.append(values["distance"])
        speed_list.append(values["speed"])
        true_anomaly_list.append(values["true_anomaly"])
        raan_list.append(values["raan"])
        arg_perigee_list.append(values["argument_perigee"])
        inclination_list.append(values["inclination"])

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


    # Plotting

    # true anomaly
    plt.figure(figsize=(9 , 6))
    ts_hour = ts/3600
    plt.plot(ts_hour, orbital_elements_converted.theta, alpha=1, label="Calculated Value")
    plt.plot(relative_time_list, true_anomaly_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("True Anomaly [deg]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("true_anomaly.png", bbox_inches="tight", dpi=500)

    # distance
    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, np.sqrt((orbital_elements_converted.pos[:, 0]**2) + (orbital_elements_converted.pos[:, 1]**2) + (orbital_elements_converted.pos[:, 2]**2)), label="Calculated Value")
    plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("Distance [km]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("distance.png", bbox_inches="tight", dpi=500)

    # speed
    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, np.sqrt((orbital_elements_converted.vel[:, 0]**2) + (orbital_elements_converted.vel[:, 1]**2) + (orbital_elements_converted.vel[:, 2]**2)), label="Calculated Value")
    plt.plot(relative_time_list, speed_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("Speed [km/s]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("speed.png", bbox_inches="tight", dpi=500)


    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, orbital_elements_converted.raan, label="Calculated Value")
    plt.plot(relative_time_list, raan_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("RAAN [deg]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend
    plt.savefig("raan.png", bbox_inches="tight", dpi=500)


    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, orbital_elements_converted.arg_perigee, label="Calculated Value")
    plt.plot(relative_time_list, arg_perigee_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("Argument of Perigee [deg]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("arg_perigee.png", bbox_inches="tight", dpi=500)
    # plt.plot(ts, np.sqrt((rs[:, 0]**2) + (rs[:, 1]**2) + (rs[:, 2]**2)))
    # plt.plot(ts, np.sqrt((vs[:, 0]**2) + (vs[:, 1]**2) + (vs[:, 2]**2)) *1000)
    # plt.show()
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