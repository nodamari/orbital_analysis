import datetime
import math
import pickle
from cProfile import label

from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from tle_obj import TLE
from orbital_elements import OE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter
from scipy.integrate import ode
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    with open("sc.pkl", "rb") as f:
        sc = pickle.load( f)

    time = (sc[:, 6] - sc[:, 7])/3600

    # plt.figure()
    # plt.title("RAAN")
    # plt.plot(time, sc[:, 5])
    # plt.grid(linestyle="--")
    # plt.xlabel("Time [hour]")
    # plt.ylabel("RAAN [deg]")

    # plt.figure()
    # plt.title("Argument of Perigee")
    # plt.plot(time, sc[:, 4])
    # plt.grid(linestyle="--")
    # plt.xlabel("Time [hour]")
    # plt.ylabel("AOP [deg]")

    plt.figure()
    plt.title("True Anomaly")
    plt.plot(time, sc[:,3])
    plt.grid(linestyle="--")
    plt.xlabel("Time [hour]")
    plt.ylabel("TA [deg]")

    # plt.figure()
    # plt.title("Eccentricity")
    # plt.plot(time, sc[:,1])
    # plt.grid(linestyle="--")
    # plt.xlabel("Time [hour]")
    # plt.ylabel("")



    plt.show()

"""
if __name__ == '__main__':
    with open("states.pkl", "rb") as f:
        j2 = pickle.load(f)

    with open("states_noj2.pkl", "rb") as f:
        no_j2 = pickle.load(f)


    vel_j2 = j2["vs"]
    vel_noj2 = no_j2["vs"]
    ts = j2["ts"]
    ts_hour = ts / 3600
    plt.figure(figsize=(9 , 6))

    plt.plot(ts_hour, vel_j2[:, 0], label="J2")
    plt.plot(ts_hour, vel_noj2[:, 0], label="No J2")
    plt.xlabel("Time [hour]")
    plt.ylabel("Vel X [km]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()

    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, vel_j2[:, 1], label="J2")
    plt.plot(ts_hour, vel_noj2[:, 1], label="No J2")
    plt.xlabel("Time [hour]")
    plt.ylabel("Vel Y [km]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()


    plt.figure(figsize=(9 , 6))
    plt.plot(ts_hour, vel_j2[:, 2], label="J2")
    plt.plot(ts_hour, vel_noj2[:, 2], label="No J2")
    plt.xlabel("Time [hour]")
    plt.ylabel("Vel Z [km]")
    plt.xlim(xmin=min(ts_hour)-1, xmax = max(ts_hour)+1)
    plt.grid(linestyle="--")
    plt.legend()

    plt.show()

"""