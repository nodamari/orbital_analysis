from scipy.optimize import fsolve
import numpy as np
import math
import spiceypy as spice
import datetime
earth_mu = 398600.4418 # km^3 / s^2


def eccentric_anomaly(E, M, eccentricity):
    M = np.radians(M)
    return E - (eccentricity*np.sin(E)) - M


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


def rotation313(arr, s, t, p, a):
    s = np.deg2rad(s) # psi , RAAN
    t = np.deg2rad(t) # theta, inclination
    p = np.deg2rad(p) # phi, arg peri
    a = np.deg2rad(a)

    cos_s = np.cos(s)
    sin_s = np.sin(s)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_p = np.cos(p)
    sin_p = np.sin(p)



    mat_1 = (cos_s * cos_p) - (sin_s * sin_p * cos_t)
    mat_2 = -(cos_s * sin_p) - (sin_s * cos_t * cos_p)
    mat_3 = sin_s * sin_t
    mat_4 = (sin_s * cos_p) + (cos_s * sin_p* cos_t)
    mat_5 = -(sin_s * sin_p) + (cos_s * cos_t * cos_p)
    mat_6 = -cos_s * sin_t
    mat_7 = sin_t * sin_p
    mat_8 = sin_t * cos_p
    mat_9 = cos_t

    rot_arr = np.array(([mat_1, mat_2, mat_3], [mat_4, mat_5, mat_6], [mat_7, mat_8, mat_9]))

    return np.matmul(rot_arr, arr)


class TLE:
    def __init__(self, input):
        self.norad_cat_id = input[1]
        self.epoch = float(input[3])
        self.datetime = self.datetime(self.epoch)
        self.bstar = float(input[4])
        self.inclination = float(input[12])
        self.right_ascension = float(input[13])
        self.eccentricity = float('0.' + input[14])
        self.arg_perigee = float(input[15])
        self.mean_anomaly = float(input[16])
        self.mean_motion = float(input[17])
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
        self.spice_pos_arr, self.spice_vel_arr = self.spice_conic()


    def datetime(self, epoch):
        year = int("20" + str(epoch)[0:2])
        day = epoch % 1000

        epoch_datetime = datetime.datetime(year =year, month=1, day=1) + datetime.timedelta(days=day-1)

        return epoch_datetime


    def true_anomaly(self):  # true anomaly in degrees
        E = fsolve(eccentric_anomaly, 2, args=(self.mean_anomaly, self.eccentricity))
        self.theta = 2*np.arctan((np.sqrt((1+self.eccentricity) / (1 - self.eccentricity)))*(np.tan(E/2)))
        if self.theta < 0:
            self.theta = 2*math.pi + self.theta
        self.theta = np.degrees(self.theta)[0]
        return self.theta

    def semi_major_axis(self):  # semi-major axis in km
        p = 1/self.mean_motion*24*60*60  # convert mean motion to orbital period in seconds
        self.a = (p**2*earth_mu/4.0/np.pi**2)**(1/3.0)
        #self.a = (p**2*(earth_mu/(4*math.pi**2)))**(1/3)

        #self.a = (earth_mu / (self.mean_motion**2))**(1/3)
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
        self.orbit = rotation313(pf_orbit, self.right_ascension, self.inclination, self.arg_perigee, self.theta)

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
        self.h = rotation313(self.h, self.right_ascension, self.inclination, self.arg_perigee, self.theta)
        return self.eta, self.h

    def object_pos(self):
        """
        Takes in the true anomaly to determine the x and y values of the object in the orbit in the perifocal frame
        and then rotates the point using the 313 rotation.
        """
        self.pos_arr = np.zeros((3, 1))
        r = fsolve(distance, -self.a, args=(self.theta, self.a, self.b))
        r = r[0]  # take it out of the array
        if r < 0:
            r = -r
        t = np.deg2rad(self.theta)

        d = np.sqrt(r**2 + self.c**2 - (2*r*self.c*np.cos(np.deg2rad(360-self.theta))))
        x = d*np.cos(t)
        y = d*np.sin(t)

        pos_arr = np.array(([x], [y], [0]))

        self.pos_arr = rotation313(pos_arr, self.right_ascension, self.inclination, self.arg_perigee, self.theta)
        return self.pos_arr

    def object_vel(self):
        pos_scalar = np.linalg.norm(self.pos_arr) # distance from Earth center to object [km]
        vel_scalar = np.sqrt(2 * (self.eta + (earth_mu / pos_scalar)))  # [km/s]

        vel_scalar = np.sqrt(earth_mu * ((2/pos_scalar) - (1/self.a)))
        x_dot = (earth_mu / np.linalg.norm(self.h)) * -np.sin(np.deg2rad(self.theta)) #- pos_scalar*np.sin(np.deg2rad(self.theta))
        y_dot = (earth_mu / np.linalg.norm(self.h)) * (self.eccentricity + np.cos(np.deg2rad(self.theta))) # pos_scalar * np.cos(np.deg2rad(self.theta))

        vel_arr = np.array(([x_dot], [y_dot], [0]))

        #unit_vel_arr = vel_arr / np.linalg.norm(vel_arr)


        # vel_arr = vel_scalar * unit_vel_arr

        self.vel_arr = rotation313(vel_arr, self.right_ascension, self.inclination, self.arg_perigee, self.theta)
        return self.vel_arr


    def spice_conic(self):
        state = spice.conics([self.perigee, self.eccentricity, np.deg2rad(self.inclination),
                                           np.deg2rad(self.right_ascension), np.deg2rad(self.arg_perigee),
                                           np.deg2rad(self.mean_anomaly), self.epoch, earth_mu], self.epoch)
        pos_arr = state[0:3]
        vel_arr = state[3:] * 1.005

        self.spice_pos_arr = pos_arr
        self.spice_vel_arr = vel_arr
        return self.spice_pos_arr, self.spice_vel_arr