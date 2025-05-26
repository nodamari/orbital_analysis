import math
import pickle
import os
from bisect import bisect_left
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tle_obj import TLE
from space_object import SpaceObject


earth_radius = 6378.135 # km
earth_mu = 398600.4418 # km^3 / s^2
J2 = 1.08262668e-3

def take_closest(list, num):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(list, num)
    if pos == 0:
        return list[0]
    if pos == len(list):
        return list[-1]
    before = list[pos - 1]
    after = list[pos]
    if after - num < num - before:
        return after
    else:
        return before

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

def diffy_q(t, y):
    # unpacking the elements in state
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])
    # norm of radius vector
    norm_r = np.linalg.norm(r)

    r2 = norm_r**2
    z2 = r[2]**2
    # Earth's gravitational acceleration
    ax, ay, az = -(r*earth_mu)/(norm_r**3)

    _a = np.linalg.norm(np.array((ax, ay, az)))

    # J2 correction
    r_J2 = 1.5 * J2 * ((earth_mu * earth_radius**2) / norm_r**4)
    ax_J2 = r_J2 * rx / norm_r * (5 * z2 / r2 - 1)
    ay_J2 = r_J2 * ry / norm_r * (5 * z2 / r2 - 1)
    az_J2 = r_J2 * rz / norm_r * (5 * z2 / r2 - 3)

    _aj2 = np.linalg.norm(np.array((ax_J2, ay_J2, az_J2)))
    # atmospheric drag
    # drag_x = air_density_ratio(norm_r) * bstar / earth_radius * (vx ** 2)
    # drag_y = air_density_ratio(norm_r) * bstar / earth_radius * (vy ** 2)
    # drag_z = air_density_ratio(norm_r) * bstar / earth_radius * (vz ** 2)

    ax += ax_J2 #+ drag_x
    ay += ay_J2 #+ drag_y
    az += az_J2 #+ drag_z

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

    # make sure the list is sorted by the epoch in increasing order, source: https://stackoverflow.com/questions/17555218
    tle_list.sort(key=lambda x: float(x[3]))

    # remove duplicates, source: https://stackoverflow.com/questions/2213923
    tle_list = list(tle_list for tle_list, _ in itertools.groupby(tle_list))

    return tle_list

def plot_output(simulation_time, simulation_data, sim_relative_time, tle_data, ylabel, title, save_name ):
    plt.figure(figsize=(10,6))
    plt.plot(simulation_time, simulation_data, label="Calculated Data")
    plt.plot(sim_relative_time, tle_data, linestyle="None", marker="*", label="Database Data")
    plt.title(title)
    plt.xlabel("Time [hours]")
    plt.ylabel(ylabel)
    plt.xlim(xmin=min(simulation_time)-1, xmax = max(simulation_time)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig(os.path.join("plots", f"{save_name}.png"), bbox_inches="tight", dpi=500)


if __name__ == '__main__':
    norad_id = 55473
    # query TLEs with query_tles.py first
    with open(f"raw_tles_{norad_id}.pkl", "rb") as f:
        raw_tles = pickle.load(f)

    # clean up the raw TLE inputs to list of TLEs
    tle_list = read_multiple_tles(raw_tles)

    # TLE object list
    tle_obj_list = []
    for tle in tle_list:
        tle_object = TLE(tle)
        tle_obj_list.append(tle_object)

    # store the TLE elements and values into a list to plot later
    relative_time_list = []
    distance_list = []
    speed_list = []
    angular_momentum_list = []
    true_anomaly_list = []
    raan_list = []
    arg_perigee_list = []
    inclination_list = []
    eccentricity_list = []

    relative_time = -1
    for t in range(len(tle_obj_list)):
        tle_obj = tle_obj_list[t]
        if t == 0:
            relative_time = 0
        else:
            relative_epoch = tle_obj_list[t].datetime - tle_obj_list[0].datetime
            relative_time = ((relative_epoch.days * 24) + ((relative_epoch.seconds + (relative_epoch.microseconds/1000000)) / 3600)) # [hour]

        distance = np.linalg.norm(tle_obj.pos_arr)
        speed = np.linalg.norm(tle_obj.vel_arr)
        h = np.linalg.norm(tle_obj.h)
        relative_time_list.append(relative_time)

        distance_list.append(distance)
        speed_list.append(speed)
        angular_momentum_list.append(h)
        true_anomaly_list.append(tle_obj.theta)
        raan_list.append(tle_obj.right_ascension)
        arg_perigee_list.append(tle_obj.arg_perigee)
        inclination_list.append(tle_obj.inclination)
        eccentricity_list.append(tle_obj.eccentricity)

    # intial position and velocity vectors
    sat = tle_obj_list[0]
    sat_next = tle_obj_list[4]
    r0 = [sat.pos_arr[0, 0], sat.pos_arr[1, 0], sat.pos_arr[2, 0]]
    v0 = [sat.vel_arr[0, 0], sat.vel_arr[1, 0], sat.vel_arr[2, 0]]


    # time span
    d_epoch = sat_next.epoch - sat.epoch
    tspan = d_epoch*24*60*60
    y0 = r0 + v0

    # instantiate SpaceObject here with states
    config = {"state_vector": r0 + v0,
              "tspan": tspan,
              "J2": True}
    so = SpaceObject(config)


    # Plotting orbital elements and other states
    if not os.path.exists("plots"):
        os.mkdir("plots")

    ts_hour = so.ts / 3600

    # distance
    plot_output(ts_hour, so.r_norm,
                relative_time_list, distance_list,
                "Distance [km]", "Distance", "distance")

    # speed
    plot_output(ts_hour, so.v_norm,
                relative_time_list, speed_list,
                "Speed [km/s]", "Speed", "speed")

    # specific angular momentum
    plot_output(ts_hour, so.orbital_elements[:,0],
                relative_time_list, angular_momentum_list,
                "H [km^2/s]", "Specific Angular Momentum", "angular_momentum")

    # eccentricity
    plot_output(ts_hour, so.orbital_elements[:,1],
                relative_time_list, eccentricity_list,
                "Eccentricity", "Eccentricity", "eccentricity")

    # inclination
    plot_output(ts_hour, so.orbital_elements[:,2],
                relative_time_list, inclination_list,
                "Inclination [deg]", "Inclination", "inclination")

    # true anomaly
    plot_output(ts_hour, so.orbital_elements[:,3],
                relative_time_list, true_anomaly_list,
                "True Anomaly [deg]", "True Anomaly", "true_anomaly")

    # argument of perigee
    plot_output(ts_hour, so.orbital_elements[:,4],
                relative_time_list, arg_perigee_list,
                "Argument of Perigee [deg]", "Argument of Perigee", "arg_perigee")

    # RAAN
    plot_output(ts_hour, so.orbital_elements[:,5],
                relative_time_list, raan_list,
                "RAAN [deg]", "RAAN", "raan")


    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sat_path = sat.orbit
    sat_next_path = sat_next.orbit
    current_pos = sat.spice_pos_arr
    next_pos = sat_next.spice_pos_arr
    equator = circle(earth_radius)

    # plot orbit of object and equator
    plt.plot(sat_path[0], sat_path[1], sat_path[2], color='deepskyblue', linestyle='solid', linewidth=2,  label='sat')
    ax.plot(sat_next_path[0], sat_next_path[1], sat_next_path[2], color='hotpink', linestyle='solid', linewidth=2, label='sat {0:.2f} hours'.format(d_epoch*24))
    plt.plot(current_pos[0], current_pos[1], current_pos[2], marker='o', markerfacecolor='deepskyblue', markeredgecolor='deepskyblue', label='sat current position')
    plt.plot(next_pos[0], next_pos[1], next_pos[2], marker='o', markerfacecolor='hotpink', markeredgecolor='hotpink', label='sat next position')
    plt.plot(so.rs[:, 0][-1], so.rs[:, 1][-1], so.rs[:, 2][-1], marker='o', markerfacecolor='blueviolet', markeredgecolor='blueviolet', label='sat numerical position')
    plt.plot(equator[0], equator[1], equator[2], color='limegreen', linestyle='dashed', label='Equator')
    ax.quiver(so.rs[0][0], so.rs[0][1], so.rs[0][2],  so.vs[0][0]*300,  so.vs[0][1]*300, so.vs[0][2]*300, color='deepskyblue')
    ax.plot(so.rs[:, 0], so.rs[:, 1], so.rs[:, 2], color='blueviolet', linestyle='solid', alpha=0.5,
        label='Numerically calculated trajectory')

    plt.legend()

    # plotting Earth
    _u, _v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    _x = earth_radius * np.cos(_u) * np.sin(_v)
    _y = earth_radius * np.sin(_u) * np.sin(_v)
    _z = earth_radius * np.cos(_v)

    # axes label
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.get_proj= lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.85, 0.85, 1, 1]))
    plt.legend()
    ax.plot_surface(_x,_y,_z, color='honeydew', label='Earth', zorder=1, alpha=0.3)

    plt.show()