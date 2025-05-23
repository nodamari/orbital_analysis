from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
import pickle
import numpy as np
from spacetrack import SpaceTrackClient
import datetime
import spacetrack.operators as op
from sgp4.api import Satrec
from sgp4.api import jday
from sgp4.api import days2mdhms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import math

if __name__ == '__main__':
    earth_mu = 398600.4418
    earth_radius = 6378.135

    with open("states.pkl", "rb") as f:
        states = pickle.load(f)


    #get username and password from userpass.txt file
    with open('userpass.txt') as f:
        contents = f.readlines()
    user = contents[0].rstrip('\n')
    password = contents[1]

    st = SpaceTrackClient(user, password)
    sat_cat_id = 44952# 25544 # ISS
    drange = op.inclusive_range(datetime.datetime(2025, 4, 15),  datetime.datetime(2025, 4, 16))
    tle = st.tle(norad_cat_id=[sat_cat_id], epoch=drange, limit=1, format='tle')

    s = tle.split("\n")[0]
    t = tle.split("\n")[1]

    satellite = Satrec.twoline2rv(s, t)

    month, day, hour, minute, second = days2mdhms(satellite.epochyr, satellite.epochdays)
    # initial_epoch = datetime.datetime(satellite.epochyr+2000, 1, 1) + datetime.timedelta(days=satellite.epochdays - 1)
    #
    # jd, fr  = jday(initial_epoch.year, initial_epoch.month, initial_epoch.day, initial_epoch.hour,
    #                initial_epoch.minute, initial_epoch.second+1)

    microsecond = int((second % 1) * 1000)
    second = int(math.floor(second))
    initial_epoch = datetime.datetime(satellite.epochyr+2000, month, day, hour, minute, second, microsecond)


    rs = []
    vs = []
    true_anom = []
    time = []
    for i in range(0, 500):
        epoch = initial_epoch + datetime.timedelta(seconds=50*i)
        second = epoch.second + epoch.microsecond/1000
        jd, fr  = jday(epoch.year, epoch.month, epoch.day, epoch.hour,
                      epoch.minute, second)

        e, r, v = satellite.sgp4(jd, fr)

        true_anom.append(satellite.mo)
        rs.append(r)
        vs.append(v)
        time.append((i*50)/ 3600)


        # h = np.cross(r, v)
        # vh_cross = np.cross(v, h) / earth_mu
        #
        # r_normalized = r / np.linalg.norm(r)
        #
        # ecc = vh_cross - r_normalized


    plt.figure()
    rv_dot = np.vecdot(np.array((rs)), np.array((vs)))
    plt.plot(time, rv_dot)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    #
    #
    # plt.legend()
    #
    # # plotting Earth
    # _u, _v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    # _x = earth_radius * np.cos(_u) * np.sin(_v)
    # _y = earth_radius * np.sin(_u) * np.sin(_v)
    # _z = earth_radius * np.cos(_v)
    #
    # # axes label
    # ax.set_xlabel('X [km]')
    # ax.set_ylabel('Y [km]')
    # ax.set_zlabel('Z [km]')
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.85, 0.85, 1, 1]))
    # plt.legend()
    # ax.plot_surface(_x, _y, _z, color='honeydew', label='Earth', zorder=1, alpha=0.3)

    plt.show()

