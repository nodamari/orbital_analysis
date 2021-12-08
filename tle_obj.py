from scipy.optimize import fsolve
import numpy as np
import math

earth_mu = 398600.0 # km^3 / s^2


def eccentric_anomaly(E, M, eccentricity):
    M = np.radians(M)
    return E - (eccentricity*np.sin(E)) - M


class TLE:
    def __init__(self, input):
        self.norad_cat_id = input[1]
        self.epoch = float(input[3])
        self.inclination = float(input[10])
        self.right_ascension = float(input[11])
        self.eccentricity = float('0.' + input[12])
        self.arg_perigee = float(input[13])
        self.mean_anomaly = float(input[14])
        self.mean_motion = float(input[15])

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

    def print(self):
        print(self.eccentricity, self.a, self.ap_peri)
