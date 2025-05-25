"""
This script requires tle_obj.py and  userpass.txt files to run.
tle_obj.py is a class object file.
userpass.txt is a text file containing the username for space-track.org in the first line and the password in the
second line.
"""
import datetime
import math
import pickle
from bisect import bisect_left

from scipy.cluster.hierarchy import correspond
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from tle_obj import TLE
from orbital_elements import OE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter
from scipy.integrate import ode, solve_ivp
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.preprocessing import normalize
import spiceypy as spice



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

def fun(S, t, mu):
    rx, ry, rz, vx, vy, vz = S

    r = np.array([rx, ry, rz])

    # norm of radius vector
    norm_r = np.linalg.norm(r)

    # Earth's gravitational acceleration
    ax, ay, az = -(r * mu) / (norm_r ** 3)

    return [vx, vy, vz, ax, ay, az]


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

    return tle_list


if __name__ == '__main__':
    norad_id = 55473
    # query TLEs with query_tles.py first
    with open(f"raw_tles_{norad_id}.pkl", "rb") as f:
        raw_tles = pickle.load(f)

    tle_list = read_multiple_tles(raw_tles)
    tle_dict = {}

    first_epoch = -1
    for tle in tle_list:
        tle_object = TLE(tle)
        tle_dict[tle_object.epoch] = {}
        tle_dict[tle_object.epoch]["object"] = tle_object
        tle_dict[tle_object.epoch]["argument_perigee"] = tle_object.arg_perigee
        tle_dict[tle_object.epoch]["eccentricity"] = tle_object.eccentricity
        tle_dict[tle_object.epoch]["true_anomaly"] = tle_object.true_anomaly()
        tle_dict[tle_object.epoch]["raan"] = tle_object.right_ascension
        tle_dict[tle_object.epoch]["inclination"] = tle_object.inclination
        tle_dict[tle_object.epoch]["pos_vec"] = tle_object.pos_arr
        tle_dict[tle_object.epoch]["vel_vec"] = tle_object.vel_arr
        tle_dict[tle_object.epoch]["spice_pos_vec"] = tle_object.spice_pos_arr
        tle_dict[tle_object.epoch]["spice_vel_vec"] = tle_object.spice_vel_arr
        tle_dict[tle_object.epoch]["distance"] = np.linalg.norm(tle_object.pos_arr)
        tle_dict[tle_object.epoch]["speed"] = np.linalg.norm(tle_object.vel_arr)
        tle_dict[tle_object.epoch]["h_vec"] = tle_object.h

        year = int("20" + str(tle_object.epoch)[0:2])
        days_into_year = tle_object.epoch % 1000
        tle_dict[tle_object.epoch]["datetime_obj"] = datetime.datetime(year=year, month=1, day=1) + datetime.timedelta(days=days_into_year-1)

        # equivalent sim time
        if len(tle_dict) == 1:
            first_epoch = tle_dict[tle_object.epoch]["datetime_obj"]
            tle_dict[tle_object.epoch]["relative_time"] = 0
        else:
            relative_epoch = tle_dict[tle_object.epoch]["datetime_obj"] - first_epoch
            tle_dict[tle_object.epoch]["relative_time"] = ((relative_epoch.days * 24) + ((relative_epoch.seconds + (relative_epoch.microseconds/1000000)) / 3600)) * 3600 # [sec]


    relative_time_list = []
    distance_list = []
    speed_list = []
    true_anomaly_list = []
    raan_list = []
    arg_perigee_list = []
    inclination_list = []
    eccentricity_list = []
    for epoch, values in tle_dict.items():
        relative_time_list.append(values["relative_time"])
        distance_list.append(values["distance"])
        speed_list.append(values["speed"])
        true_anomaly_list.append(values["true_anomaly"])
        raan_list.append(values["raan"])
        arg_perigee_list.append(values["argument_perigee"])
        inclination_list.append(values["inclination"])
        eccentricity_list.append(values["eccentricity"])

    # intial position and velocity vectors

    tle_dict_keys = list(tle_dict.keys())

    sat = tle_dict[tle_dict_keys[0]]["object"]
    sat_next = tle_dict[tle_dict_keys[4]]["object"]
    r0 = [sat.pos_arr[0, 0], sat.pos_arr[1, 0], sat.pos_arr[2, 0]]
    v0 = [sat.vel_arr[0, 0], sat.vel_arr[1, 0], sat.vel_arr[2, 0]]

    # r0 = [-2384.46, 5729.01, 3050.46]
    # v0 = [-7.36138, -2.98997, 1.64354]
    # r0 = [5659.03, 6533.74, 3270.15]
    # v0 = [-3.8797, 5.11565, -2.2397]
    # time span
    d_epoch = sat_next.epoch - sat.epoch
    tspan = d_epoch*24*60*60
    print("T Span: ", tspan/3600, " hours" )

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


    ode_solution = solve_ivp(fun=diffy_q, t_span=(0, tspan), y0 = y0, method='LSODA', dense_output=False, atol=1e-6, rtol=1e-7)
    ys = ode_solution.y
    rs = ys[0:3, :].T
    vs = ys[3:, :].T
    ts = ode_solution.t
    print("Length of Solution ", len(ts))

    # step = 1 # second step because first step defined as initial contidions
    #
    # # intialize solver
    # solver = ode(diffy_q)
    # solver.set_integrator('dop853')
    # solver.set_initial_value(y0, 0)
    # #solver.set_f_params(earth_mu, sat.bstar)
    #
    # # propagate orbit
    # while solver.successful() and step < n_steps:
    #     solver.integrate(solver.t + dt)
    #     ts[step] = solver.t
    #     ys[step] = solver.y
    #     step +=1
    # rs = ys[:, :3]
    # vs = ys[:, 3:]


    equator = circle(earth_radius)

    orbital_elements_converted = OE(rs, vs)

    hu = (np.linalg.norm(orbital_elements_converted.h, axis=1) **2) / (earth_mu**2)

    e = np.sqrt(1 + hu * (orbital_elements_converted.v**2 - (2*earth_mu / orbital_elements_converted.r)))

    states = {}
    states["rs"] = rs
    states["vs"] = vs
    states["ts"] = ts
    states["initial_epoch"] = sat.epoch
    states["rp"] = sat.perigee
    states["ecc"] = sat.eccentricity
    states["inc"] = sat.inclination
    states["lnode"] = sat.right_ascension
    states["argp"] = sat.arg_perigee
    states["m0"] = sat.mean_anomaly
    states["e"] = orbital_elements_converted.e_vec

    with open("states.pkl", "wb") as f:
        pickle.dump(states, f)


    # for key, value in tle_dict.items():
    #     tle = tle_dict[key]
    #     relative_time = tle["relative_time"]
    #     pos_vec = tle["pos_vec"]
    #     vel_vec = tle["vel_vec"]
    #
    #     closest_time = take_closest(list(ts), relative_time)
    #     closest_time_idx = int(closest_time/dt)
    #     corresponding_pos = rs[closest_time_idx]
    #
    #     diff = pos_vec[:,0] - corresponding_pos
    #     print(pos_vec[:,0])
    #     print(corresponding_pos)
    #     print(np.linalg.norm(diff))
    #     print()



    # Plotting
    ts_hour = ts / 3600
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, orbital_elements_converted.h[:, 0], label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Hx [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)
    #
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, orbital_elements_converted.h[:, 1], label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Hy [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)
    #
    #
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, orbital_elements_converted.h[:, 2], label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Hz [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)


    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, np.linalg.norm(orbital_elements_converted.h, axis=1), label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("H [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)


    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, np.linalg.norm(orbital_elements_converted.h, axis=1), label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("H [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)


    # distance
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, orbital_elements_converted.pos[:, 0], label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("X [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)
    #
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, orbital_elements_converted.pos[:, 1], label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Y [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)
    #
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, orbital_elements_converted.pos[:, 2], label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Z [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)
    #
    # plt.figure(figsize=(9 , 6))
    # ts_hour = ts / 3600
    # plt.plot(ts_hour, np.linalg.norm(rs, axis=1), label="Calculated Value")
    # #plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Distance [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("distance.png", bbox_inches="tight", dpi=500)

    # # # true anomaly
    plt.figure(figsize=(9 , 6))
    ts_hour = ts/3600
    relative_time_list = np.array((relative_time_list)) / 3600
    plt.plot(ts_hour, orbital_elements_converted.theta, alpha=1, label="Calculated Value")
    plt.plot(relative_time_list, true_anomaly_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("True Anomaly [deg]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("true_anomaly.png", bbox_inches="tight", dpi=500)

    # # distance
    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, np.sqrt((rs[:, 0]**2) + (rs[:, 1]**2) + (rs[:, 2]**2)), label="Calculated Value")
    plt.plot(relative_time_list, distance_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("Distance [km]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("distance.png", bbox_inches="tight", dpi=500)
    #
    # # speed
    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, np.sqrt((orbital_elements_converted.vel[:, 0]**2) + (orbital_elements_converted.vel[:, 1]**2) + (orbital_elements_converted.vel[:, 2]**2)), label="Calculated Value")
    # plt.plot(relative_time_list, speed_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Speed [km/s]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("speed.png", bbox_inches="tight", dpi=500)


    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, np.sqrt((orbital_elements_converted.vel[:, 0]**2) + (orbital_elements_converted.vel[:, 1]**2) + (orbital_elements_converted.vel[:, 2]**2)), label="Calculated Value")
    # plt.plot(relative_time_list, speed_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Speed [km/s]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("speed.png", bbox_inches="tight", dpi=500)

    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, orbital_elements_converted.vel[:, 0], label="X")
    # plt.plot(ts_hour, orbital_elements_converted.vel[:, 1], label="Y")
    # plt.plot(ts_hour, orbital_elements_converted.vel[:, 2], label="Z")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Vel [km/s]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("speed.png", bbox_inches="tight", dpi=500)


    #
    # # RAAN
    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, orbital_elements_converted.raan, label="Calculated Value")
    plt.plot(relative_time_list, raan_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("RAAN [deg]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("raan.png", bbox_inches="tight", dpi=500)
    # # #
    # # # # argument of perigee
    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, orbital_elements_converted.arg_perigee, label="Calculated Value")
    plt.plot(relative_time_list, arg_perigee_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("Argument of Perigee [deg]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("arg_perigee.png", bbox_inches="tight", dpi=500)
    #
    # inclination
    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, orbital_elements_converted.i, label="Calculated Value")
    # plt.plot(relative_time_list, inclination_list, linestyle="None", marker="*", label="Database Value")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Inclination [deg]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("inclination.png", bbox_inches="tight", dpi=500)
    #
    # # eccentricity
    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour,e, label="Calculated Value")
    plt.plot(relative_time_list, eccentricity_list, linestyle="None", marker="*", label="Database Value")
    plt.xlabel("Time [hours]")
    plt.ylabel("Eccentricity")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()
    plt.savefig("eccentricity.png", bbox_inches="tight", dpi=500)


    # xyz eccentricity
    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, orbital_elements_converted.e_vec[:,0], label="X")
    # plt.plot(ts_hour, orbital_elements_converted.e_vec[:, 1], label="Y")
    # plt.plot(ts_hour, orbital_elements_converted.e_vec[:, 2], label="Z")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Eccentricity")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour[30])+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("eccentricity.png", bbox_inches="tight", dpi=500)
    #
    # plt.figure(figsize=(9 , 6))
    # plt.plot(ts_hour, orbital_elements_converted.pos[:,0]/orbital_elements_converted.r[:], label="X")
    # plt.plot(ts_hour, orbital_elements_converted.pos[:, 1]/orbital_elements_converted.r[:], label="Y")
    # plt.plot(ts_hour, orbital_elements_converted.pos[:, 2]/orbital_elements_converted.r[:], label="Z")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("Position [km]")
    # plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour[30])+1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("eccentricity.png", bbox_inches="tight", dpi=500)
    #
    # vh_cross = np.cross(orbital_elements_converted.vel, orbital_elements_converted.h) / earth_mu
    #
    # plt.figure(figsize=(9, 6))
    # plt.plot(ts_hour, vh_cross[:, 0], label="X")
    # plt.plot(ts_hour, vh_cross[:, 1], label="Y")
    # plt.plot(ts_hour, vh_cross[:, 2], label="Z")
    # plt.xlabel("Time [hours]")
    # plt.ylabel("V X H [km^3/s^2]")
    # plt.xlim(xmin=min(ts_hour) - 1, xmax=max(ts_hour[30]) + 1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("eccentricity.png", bbox_inches="tight", dpi=500)



    # angle between v and h

    # angles = []
    # for i in range(500):
    #     angles.append(np.dot(orbital_elements_converted.pos[i], orbital_elements_converted.vel[i]) / orbital_elements_converted.r[i])
    #     #angles.append(angle(orbital_elements_converted.pos[i], orbital_elements_converted.vel[i]))
    #     #angles.append(np.linalg.norm(np.cross(orbital_elements_converted.vel[i], orbital_elements_converted.h[i]))/earth_mu)
    #
    # r = []
    # for j in range(500):
    #     r.append(np.linalg.norm(orbital_elements_converted.pos[j] / orbital_elements_converted.r[j]))
    # plt.figure(figsize=(9, 6))
    # plt.plot(ts_hour[0:500], angles, label="e1")
    # # plt.plot(ts_hour, r, label="e2")
    # #plt.plot(ts_hour[0:500], np.array(angles) - np.array(r), label='diff')
    # plt.xlabel("Time [hours]")
    # plt.ylabel("V and H vecot angle")
    # plt.xlim(xmin=min(ts_hour[0:500]) - 1, xmax=max(ts_hour[0:500]) + 1)
    # plt.grid(linestyle="--")
    # plt.legend()
    # plt.savefig("eccentricity.png", bbox_inches="tight", dpi=500)
    #
    # # PLOTTING #
    #
    # # initialize the figure
    # # writer = PillowWriter(fps=20)
    # #
    # #
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sat_path = sat.orbit
    sat_next_path = sat_next.orbit
    current_pos = sat.spice_pos_arr
    next_pos = sat_next.spice_pos_arr
    # plot orbit of object and equator
    plt.plot(sat_path[0], sat_path[1], sat_path[2], color='deepskyblue', linestyle='solid', linewidth=2,  label='sat')
    ax.plot(sat_next_path[0], sat_next_path[1], sat_next_path[2], color='hotpink', linestyle='solid', linewidth=2, label='sat {0:.2f} hours'.format(d_epoch*24))
    plt.plot(current_pos[0], current_pos[1], current_pos[2], marker='o', markerfacecolor='deepskyblue', markeredgecolor='deepskyblue', label='sat current position')
    plt.plot(next_pos[0], next_pos[1], next_pos[2], marker='o', markerfacecolor='hotpink', markeredgecolor='hotpink', label='sat next position')
    plt.plot(rs[:, 0][-1], rs[:, 1][-1], rs[:, 2][-1], marker='o', markerfacecolor='blueviolet', markeredgecolor='blueviolet', label='sat numerical position')



    #ax.plot(rs[1650:, 0], rs[1650:, 1], rs[1650:, 2], color='blueviolet', linestyle='solid', alpha=0.5, label='Numerically calculated trajectory')
    plt.plot(equator[0], equator[1], equator[2], color='limegreen', linestyle='dashed', label='Equator')
    #ax.quiver(sat.pos_arr[0], sat.pos_arr[1], sat.pos_arr[2],  sat.vel_arr[0]*300,  sat.vel_arr[1]*300, sat.vel_arr[2]*300, color='deepskyblue')

    # ecc_multiplier = earth_radius/orbital_elements_converted.e[0]
    # ax.quiver(0, 0, 0,
    #           orbital_elements_converted.e_vec[0][0]*ecc_multiplier, orbital_elements_converted.e_vec[0][1]*ecc_multiplier, orbital_elements_converted.e_vec[0][2]*ecc_multiplier, color='orange')
    #
    # ax.quiver(0, 0, 0,
    #           orbital_elements_converted.e_vec[10][0]*ecc_multiplier, orbital_elements_converted.e_vec[10][1]*ecc_multiplier, orbital_elements_converted.e_vec[10][2]*ecc_multiplier, color='orange')
    #
    # ax.quiver(0, 0, 0,
    #           orbital_elements_converted.e_vec[20][0]*ecc_multiplier, orbital_elements_converted.e_vec[20][1]*ecc_multiplier, orbital_elements_converted.e_vec[20][2]*ecc_multiplier, color='orange')


    ax.quiver(rs[0][0], rs[0][1], rs[0][2],  vs[0][0]*300,  vs[0][1]*300, vs[0][2]*300, color='deepskyblue')

    vh_cross = np.cross(orbital_elements_converted.vel, orbital_elements_converted.h)/earth_mu
    r_normalized = normalize(rs, axis=1, norm='l1')

    e_diff = vh_cross - r_normalized
    j = 24
    # ax.quiver(0,0,0, vh_cross[j, 0]*8000, vh_cross[j, 1]*8000, vh_cross[j, 2]*8000, color='darkorange')
    # ax.quiver(0, 0, 0, orbital_elements_converted.e1[j, 0] * 7000, orbital_elements_converted.e1[j, 1] * 7000, orbital_elements_converted.e1[j, 2] * 7000, color='orange')
    #
    # ax.quiver(0, 0, 0, rs[j, 0], rs[j, 1], rs[j, 2], color='red')
    # ax.quiver(0, 0, 0, orbital_elements_converted.e2[j, 0]*20000, orbital_elements_converted.e2[j, 1]*20000, orbital_elements_converted.e2[j, 2]*20000, color='maroon')
    #
    # ax.quiver(0, 0, 0, e_diff[j, 0]*10000, e_diff[j, 1]*10000, e_diff[j, 2]*10000, color='green')
    #
    #
    # ax.quiver(0, 0, 0, orbital_elements_converted.e_vec[j, 0] * 20000000, orbital_elements_converted.e_vec[j, 1] * 20000000, orbital_elements_converted.e_vec[j, 2] * 20000000, color='limegreen')

    #ax.quiver(sat.pos_arr[0], sat.pos_arr[1], sat.pos_arr[2],  sat.vel_arr[0]*300,  sat.vel_arr[1]*300, sat.vel_arr[2]*300, color='deepskyblue')

    ax.plot(rs[:, 0], rs[:, 1], rs[:, 2], color='blueviolet', linestyle='solid', alpha=0.5,
        label='Numerically calculated trajectory')

    #plt.plot(rs[j, 0], rs[j, 1], rs[j, 2], marker='o', markerfacecolor='red', markeredgecolor='red', label=f'{j}')
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