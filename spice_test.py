import spiceypy as spice
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt


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




if __name__ == '__main__':
   with open("states.pkl", "rb") as f:
       states = pickle.load(f)

   with open("states_noj2.pkl", "rb") as f:
       states_noj2 = pickle.load(f)

   epoch = str(states["initial_epoch"])
   year = int("20" + epoch[0:2])
   days = float(epoch[2:])


   J2000_start = datetime.datetime(2000, 1, 1, 12)
   initial_epoch = datetime.datetime(year, 1, 1) + datetime.timedelta(days=days-1)

   initial_seconds = (initial_epoch - J2000_start).total_seconds()

   earth_mu = 398600.4418

   rp_ = states["rp"]
   ecc_ = states["ecc"]
   inc_ = np.deg2rad(states["inc"])
   raan_ = np.deg2rad(states["lnode"])
   argp_ = np.deg2rad(states["argp"])
   m0_ = np.deg2rad(states["m0"])
   t0_ = initial_seconds

   rs = states["rs"]
   vs = states["vs"]
   ts = states["ts"]
   ts_hour = ts/3600

   ecc_list = []
   raan_list = []
   aop_list = []
   trueanom_list = []
   for idx in range(len(states["ts"])):
     ys = np.concatenate((states["rs"][idx], states["vs"][idx]))
     [rp, e, i, raan, argp, m0, t0, mu, nu, a, tau] = spice.oscltx(ys, initial_seconds + (50 * idx), earth_mu)
     ecc_list.append(e)
     raan_list.append(raan)
     aop_list.append(argp)
     trueanom_list.append(nu)

   plt.figure()
   # plt.plot(ts_hour, vs[:, 0])
   # plt.plot(ts_hour, vs[:, 1])
   # plt.plot(ts_hour, vs[:, 2])
   plt.plot(ts_hour, np.linalg.norm(vs, axis=1))
   plt.plot(ts_hour, np.linalg.norm(states_noj2["vs"], axis=1))
   plt.grid(linestyle="--")
   plt.xlim(xmin=min(states["ts"] / 3600) - 1, xmax=max(states["ts"] / 3600) + 1)
   plt.title("Velocity")


   plt.figure()
   # plt.plot(ts_hour, vs[:, 0])
   # plt.plot(ts_hour, vs[:, 1])
   # plt.plot(ts_hour, vs[:, 2])
   plt.plot(ts_hour, np.linalg.norm(rs, axis=1))
   plt.plot(ts_hour, np.linalg.norm(states_noj2["rs"], axis=1))
   plt.grid(linestyle="--")
   plt.xlim(xmin=min(states["ts"] / 3600) - 1, xmax=max(states["ts"] / 3600) + 1)
   plt.title("Position")

   # plt.figure()
   # plt.plot(states["ts"]/3600, ecc_list)
   # plt.title("Eccentricity")
   #
   # plt.figure()
   # plt.plot(states["ts"]/3600, raan_list)
   # plt.title("RAAN")
   #
   # plt.figure()
   # plt.plot(states["ts"]/3600, aop_list)
   # plt.title("Argument of Perigee")
   #
   # plt.figure()
   # plt.plot(states["ts"]/3600, trueanom_list)
   # plt.title("True Anomaly")

   rv_dot = np.vecdot(states["rs"], states["vs"])
   rv_dot_noj2 = np.vecdot(states_noj2["rs"], states_noj2["vs"])
   plt.figure()
   plt.plot(states["ts"]/3600, rv_dot)
   plt.plot(states["ts"] / 3600, rv_dot_noj2)
   plt.grid(linestyle="--")
   plt.xlim(xmin=min(states["ts"] / 3600) - 1, xmax=max(states["ts"] / 3600) + 1)
   plt.title("R V Dot")

   # rv_angle = []
   # for i in range(len(states["ts"])):
   #    rv_angle.append(angle(rs[i], vs[i]))
   # plt.figure()
   # plt.plot(ts/3600, rv_angle)
   plt.show()