from scipy.optimize import fsolve
import numpy as np
import math

earth_mu = 398600.0 # km^3 / s^2


def eccentric_anomaly(E, M, eccentricity):
    M = np.radians(M)
    return E - (eccentricity*np.sin(E)) - M


def angle(v1, v2):
    v1 = v1.T
    v2 = v2.T
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)

    return np.arccos(dot_product)


def distance(r, t, a, b):
    """
    :param r: The distance from the center of Earth to the object location in km
    :param t: True anomaly
    :param a: Semi-major axis
    :param b: Semi-minor axis
    :return:
    """
    t = np.deg2rad(t)
    #return (((1-(((r*np.cos(t))**2)/(a**2)))*(b**2))**(0.5)) - r*np.sin(t)
    return np.sqrt(((1-(((r*np.cos(t))**2)/(a**2)))*(b**2))) - r*np.sin(t)


def ellipse(a, b):
    x = a * np.cos(np.linspace(-np.pi, 0, 600))
    x_return = x[len(x): 0: -1]

    y = np.sqrt(((1 - ((np.multiply(x, x)) / a ** 2)) * (b ** 2)))
    y_return = -np.sqrt(((1 - ((np.multiply(x_return, x_return)) / a ** 2)) * (b ** 2)))

    x = np.concatenate((x, x_return))
    y = np.concatenate((y, y_return))
    z = np.zeros(len(x))
    path_arr = np.vstack((x, y, z))
    return path_arr


def rotation313(arr, s, t, p):
    s = np.deg2rad(s)
    t = np.deg2rad(t)
    p = np.deg2rad(p)
    # body 3: right ascension of the ascending node
    z1_rot = np.array(([np.cos(s), np.sin(s), 0], [-np.sin(s), np.cos(s), 0], [0, 0, 1]))
    # body 1: inclination
    x_rot = np.array(([1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]))
    # body 3: argument of perigee
    z2_rot = np.array(([np.cos(p), np.sin(p), 0], [-np.sin(p), np.cos(p), 0], [0, 0, 1]))

    rot1 = np.matmul(z1_rot, arr)
    rot2 = np.matmul(x_rot, rot1)
    rot3 = np.matmul(z2_rot, rot2)
    return rot3


class TLE:
    def __init__(self, input):
        self.norad_cat_id = input[1]
        self.epoch = float(input[3])
        self.bstar = float(input[4])
        self.inclination = float(input[11])
        self.right_ascension = float(input[12])
        self.eccentricity = float('0.' + input[13])
        self.arg_perigee = float(input[14])
        self.mean_anomaly = float(input[15])
        self.mean_motion = float(input[16])
        self.a = self.semi_major_axis()
        self.theta = self.true_anomaly()
        self.apogee = self.apogee_perigee()[0]
        self.perigee = self.apogee_perigee()[1]
        self.b = self.semi_minor_axis()[0]
        self.c = self.semi_minor_axis()[1]
        self.orbit = self.orbit_path()
        self.eta = self.specific_energy()[0]
        self.h = self.specific_energy()[1]
        self.pos_arr = self.object_pos()
        self.vel_arr = self.object_vel()

    def true_anomaly(self):  # true anomaly in degrees
        E = fsolve(eccentric_anomaly, 2, args=(self.mean_anomaly, self.eccentricity))
        self.theta = 2*np.arctan((np.sqrt((1+self.eccentricity) / (1 - self.eccentricity)))*(np.tan(E/2)))
        if self.theta < 0:
            self.theta = 2*math.pi + self.theta
        self.theta = np.degrees(self.theta)[0]
        return self.theta

    def semi_major_axis(self):  # semi-major axis in km
        p = (1/self.mean_motion)*24*60*60  # convert mean motion to orbital period in seconds
        self.a = (p**2*(earth_mu/(4*math.pi**2)))**(1/3)
        return self.a

    def apogee_perigee(self):
        self.ap_peri = []
        apogee = self.a + (self.eccentricity*self.a)
        perigee = self.a - (self.eccentricity*self.a)
        self.ap_peri.append(apogee)
        self.ap_peri.append(perigee)
        return self.ap_peri

    def semi_minor_axis(self):
        perigee = self.ap_peri[1]
        self.c = self.a - perigee
        self.b = np.sqrt((self.a**2) - (self.c**2))
        return self.b, self.c

    def orbit_path(self):
        """
        Creates an ellipse based on semi-major axis and semi-minor axis
        :param:
            a(float): semi-major axis
            b(float): semi-minor axis

        :return:
            path(arr): 3D array of elliptical path of an orbit
        """

        pf_orbit = ellipse(self.a, self.b)
        pf_orbit[0] -= self.c  # move the ellipse over by the distance from the center of the ellipse to the focus because Earth is defined as being at 0,0,0
        self.orbit = rotation313(pf_orbit, self.right_ascension, self.inclination, self.arg_perigee)

        return self.orbit

    def specific_energy(self):
        perigee = self.ap_peri[1]
        self.h = np.zeros((3, 1))
        h_scalar = np.sqrt(self.a*earth_mu*(1-(self.eccentricity**2)))
        self.eta = (0.5*((h_scalar**2)/(perigee**2))) - (earth_mu/perigee)
        if 90 <= self.inclination <= 180:  # account for retrograde orbits
            self.h[0][0] = 0
            self.h[1][0] = 0
            self.h[2][0] = -h_scalar
        else:
            self.h[0][0] = 0
            self.h[1][0] = 0
            self.h[2][0] = h_scalar
        self.h = rotation313(self.h, self.right_ascension, self.inclination, self.arg_perigee)
        return self.eta, self.h

    def object_pos(self):
        """
        Takes in the true anomaly to determine the x and y values of the object in the orbit in the perifocal frame
        and then rotates the point using the 313 rotation.
        """
        self.pos_arr = np.zeros((3, 1))
        r = fsolve(distance, -self.a, args=(self.theta, self.a, self.b))
        if r < 0:
            r = -r
        t = np.deg2rad(self.theta)
        x = r*np.cos(t)
        y = r*np.sin(t)

        self.pos_arr[0][0] = x
        self.pos_arr[1][0] = y
        self.pos_arr = rotation313(self.pos_arr, self.right_ascension, self.inclination, self.arg_perigee)
        return self.pos_arr

    def object_vel(self):
        pos_scalar = np.linalg.norm(self.pos_arr)
        vel_scalar = np.sqrt(2*(self.eta + (earth_mu/pos_scalar)))
        pos_mat = np.ones((3, np.shape(self.orbit)[1]))
        pos_mat[0] *= self.pos_arr[0]
        pos_mat[1] *= self.pos_arr[1]
        pos_mat[2] *= self.pos_arr[2]
        d_dist_mag = np.linalg.norm(self.orbit - pos_mat, axis=0)
        min_idx = np.argmin(d_dist_mag)
        d_arr = np.vstack((self.orbit[0,min_idx], self.orbit[1,min_idx], self.orbit[2,min_idx]))
        if 90 <= self.inclination <= 180:
            d_arr = d_arr - np.vstack((self.orbit[0,min_idx+1], self.orbit[1,min_idx+1], self.orbit[2,min_idx+1]))
        else:
            d_arr = -d_arr + np.vstack((self.orbit[0, min_idx + 1], self.orbit[1, min_idx + 1], self.orbit[2, min_idx + 1]))
        vel_unit = d_arr / np.linalg.norm(d_arr)
        self.vel_arr = vel_unit * vel_scalar

        return self.vel_arr




