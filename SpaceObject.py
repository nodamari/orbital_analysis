import numpy as np
from matplotlib.pyplot import thetagrids
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

earth_radius = 6378.135 # km
earth_mu = 398600.4418 # km^3 / s^2
J2 = 1.08262668e-3

def default_config():
    default_config = {
                      "orbital_elements": [],
                      "state_vector": [],
                      "propagate": False,
                      "tspan": 0,
                      "J2": False,
                      "state2element": True
    }

    return default_config



def state_to_orbital_elements(states):
    """
    Given a n x 6 array of states, returns a n X 6 array of orbital elements
    columns are: specific angular momentum, eccentricity, inclination, true anomaly, argument of perigee, RAAN
    """

    pos_vec = states[:, 0:3]
    vel_vec = states[:, 3:]

    distance = np.linalg.norm(pos_vec, axis=1)
    distance = distance.reshape(distance.size, 1)
    speed = np.linalg.norm(vel_vec, axis=1)
    vr = np.vecdot(pos_vec, vel_vec).reshape(speed.size, 1)/ distance

    h_vec = np.cross(pos_vec, vel_vec)
    h = np.linalg.norm(h_vec, axis=1)
    h = h.reshape(h.size, 1)
    N = np.cross([0,0,1], h_vec)
    N_norm = np.linalg.norm(N, axis=1)
    N_norm = N_norm.reshape(N_norm.size, 1)

    # inclination
    hz = h_vec[:, 2].reshape(h.size, 1)
    i = np.rad2deg(np.arccos(hz/h.reshape(h.size, 1)))
    i = i.reshape(i.size, 1)

    # eccentricity
    e1 = np.cross(vel_vec, h_vec)/ earth_mu
    e2 = np.divide(pos_vec, distance.reshape(distance.size, 1))
    e_vec = e1 - e2
    e = np.linalg.norm(e_vec, axis=1)
    e = e.reshape(e.size, 1)

    # true anomaly
    er_dot = (np.vecdot(e_vec, pos_vec))
    er_dot = er_dot.reshape(er_dot.size, 1)
    er = np.multiply(e, distance)
    theta = np.rad2deg(np.arccos(er_dot/ er))
    theta = np.where(vr < 0, 360 - theta, theta)

    # argument of perigee
    Ne_dot = np.vecdot(N, e_vec)
    Ne_dot = Ne_dot.reshape(Ne_dot.size, 1)
    Ne = np.multiply(N_norm, e)
    argp = np.rad2deg(np.arccos(Ne_dot / Ne))
    argp = np.where(e_vec[:, 2].reshape(e.size,1) < 0, 360 - argp, argp)

    # right ascension of the ascending node
    raan = np.rad2deg(np.arccos(N[:, 0].reshape(N_norm.size, 1) / N_norm))
    raan = np.where(N[:, 1].reshape(N_norm.size, 1) < 0, 360 - raan, raan)

    orbital_elements = np.concatenate((h, e, i, theta, argp, raan), axis=1)
    return orbital_elements


class SpaceObject:
    def __init__(self, config):
        self.config = default_config()

        # replace default config with input config
        for key in config.keys():
            self.config[key] = config[key]

        # make sure user has inputted some sort of state to create this object
        assert self.config["orbital_elements"] or self.config["state_vector"], "Please input an orbital elements or state vector"


        self.y0 = self.config["state_vector"]

        # propagate the orbit
        self.ys, self.ts = self.propagate_orbit(self.differential_equation, self.config["tspan"], self.y0, self.config["J2"])
        self.ys = self.ys.T
        self.ts = self.ts.reshape(self.ts.size, 1)

        if self.config["state2element"]:
            self.orbital_elements = state_to_orbital_elements(self.ys)


        print()




    def differential_equation(self, t, y, j2):
        # unpacking the elements in state
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        v = np.array([vx, vy, vz])
        # norm of radius vector
        norm_r = np.linalg.norm(r)

        r2 = norm_r ** 2
        z2 = r[2] ** 2
        # Earth's gravitational acceleration
        ax, ay, az = -(r * earth_mu) / (norm_r ** 3)

        # J2 correction
        r_J2 = 1.5 * J2 * ((earth_mu * earth_radius ** 2) / norm_r ** 4)
        ax_J2 = r_J2 * rx / norm_r * (5 * z2 / r2 - 1)
        ay_J2 = r_J2 * ry / norm_r * (5 * z2 / r2 - 1)
        az_J2 = r_J2 * rz / norm_r * (5 * z2 / r2 - 3)

        # atmospheric drag (WIP)
        # drag_x = air_density_ratio(norm_r) * bstar / earth_radius * (vx ** 2)
        # drag_y = air_density_ratio(norm_r) * bstar / earth_radius * (vy ** 2)
        # drag_z = air_density_ratio(norm_r) * bstar / earth_radius * (vz ** 2)

        if j2:
            ax += ax_J2
            ay += ay_J2
            az += az_J2

        return [vx, vy, vz, ax, ay, az]


    def propagate_orbit(self, ode_function, tspan, y0, j2):
        print("Propagating orbit...")
        ode_solution = solve_ivp(fun=ode_function, args=(j2,), t_span=(0, tspan), y0=y0, method='LSODA', dense_output=False, atol=1e-6,
                                 rtol=1e-7)
        ys = ode_solution.y
        rs = ys[0:3, :].T
        vs = ys[3:, :].T
        ts = ode_solution.t
        print("Length of Solution: ", len(ts))

        return ys, ts



if __name__ == '__main__':
    r0 = [-2384.46, 5729.01, 3050.46]
    v0 = [-7.36138, -2.98997, 1.64354]
    config = {"orbital_elements": [],
              "state_vector": r0+v0,
              "tspan": 48*3600,
              "J2": True}
    so = SpaceObject(config)





