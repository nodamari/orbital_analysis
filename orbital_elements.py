# This script is still WIP.

import numpy as np

earth_mu = 398600.4418

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

class OE:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.r = np.linalg.norm(self.pos, axis=1)
        self.v = np.linalg.norm(self.vel, axis=1)
        self.vr = np.vecdot(self.pos, self.vel)/ self.r
        self.h = np.cross(pos, vel)
        self.N = np.cross([0, 0, 1], self.h)
        self.i = self.inclination()
        self.raan = self.right_ascension_of_ascending_node()
        self.e = self.eccentricity()[0]
        self.e_vec = self.eccentricity()[1]
        self.e1 = self.eccentricity()[2]
        self.e2 = self.eccentricity()[3]
        self.arg_perigee = self.argument_of_perigee()
        self.theta = self.true_anomaly()



    def inclination(self):
        length = np.shape(self.pos)[0]
        hz = self.h[:, 2]
        self.i = np.zeros((length, 1))
        for i in range(length):
            h_norm = np.linalg.norm(self.h[i])
            self.i[i] = np.rad2deg(np.arccos((hz[i] / h_norm)))
        return self.i

    def right_ascension_of_ascending_node(self):
        length = np.shape(self.pos)[0]
        self.raan = np.zeros((length, 1))
        for i in range(length):
            Nx = self.N[i, 0]
            Ny = self.N[i, 1]
            N = np.linalg.norm(self.N[i])
            if Ny >= 0:
                self.raan[i] = np.rad2deg(np.arccos(Nx / N))
            else:
                self.raan[i] = 360 - np.rad2deg(np.arccos(Nx / N))
        return self.raan

    def eccentricity(self):
        length = np.shape(self.pos)[0]
        earth_mu = 398600.4418  # km^3 / s^2
        # e1 = np.cross(self.vel, self.h) / earth_mu
        e1 = np.zeros((length, 3))
        e2 = np.zeros((length, 3))
        e_scalar_vec = np.zeros((length, 1))
        e = np.zeros((length, 3))
        for i in range(length):

            # A = (self.v[i]**2 - (earth_mu / self.r[i]))
            # B = np.vecdot(self.pos[i], self.vel[i])
            # e_vector = (A * self.pos[i] + B * self.vel[i]) / earth_mu
            # e_normalized = np.linalg.norm(e_vector)
            #
            # e[i] = e_vector
            # e_scalar_vec[i] = e_normalized

            true_h = np.cross(self.pos[i], self.vel[i])
            vh_cross = np.cross(self.vel[i], self.h[i])
            vh_norm = np.linalg.norm(vh_cross)
            h_scalar = np.linalg.norm(true_h)
            e1[i] = np.cross(self.vel[i], self.h[i]) / earth_mu

            e1_norm = np.linalg.norm(np.cross(self.vel[i], self.h[i])) / earth_mu

            e2[i] = self.pos[i] / np.linalg.norm(self.pos[i])
            e2_norm = np.linalg.norm(self.pos[i] / np.linalg.norm(self.pos[i]))


            e1e2_diff = e1_norm - e2_norm


        e_vec = e1 - e2

        self.e_vec = e_vec
        self.e = np.linalg.norm(self.e_vec, axis=1) #e_scalar_vec #np.linalg.norm(self.e_vec, axis=1)
        self.e1 = e1
        self.e2 = e2
        return self.e, self.e_vec, self.e1, self.e2

    def argument_of_perigee(self):
        length = np.shape(self.pos)[0]
        self.arg_perigee = np.zeros((length, 1))
        for i in range(length):
            N_norm = np.linalg.norm(self.N[i])
            if self.e_vec[i,2] >= 0:
                self.arg_perigee[i] = np.rad2deg(np.arccos(np.dot(self.N[i], self.e_vec[i]) / (N_norm * self.e[i])))
            else:
                self.arg_perigee[i] = 360 - np.rad2deg(np.arccos(np.dot(self.N[i], self.e_vec[i]) / (N_norm * self.e[i])))
        return self.arg_perigee

    def true_anomaly(self):
        length = np.shape(self.pos)[0]
        self.theta = np.zeros((length, 1))

        for i in range(length):
            self.theta[i] = np.rad2deg(np.arccos(np.dot(self.e_vec[i], self.pos[i]) / (self.e[i] * self.r[i])))
            if np.dot(self.pos[i], self.vel[i]) < 0:
                self.theta[i] = 360 - self.theta[i]
        return self.theta

