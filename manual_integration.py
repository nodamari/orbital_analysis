import numpy as np
import pickle
import matplotlib.pyplot as plt

earth_mu = 398600.4418
earth_radius = 6378.135
J2 = 1.08262668e-3


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

    r = states["rs"][0]
    v = states["vs"][0]

    rx = r[0]
    ry = r[1]
    rz = r[2]

    vx = v[0]
    vy = v[1]
    vz = v[2]

    dt = 50

    nsteps= 300
    rs = np.zeros((nsteps, 3))
    vs = np.zeros((nsteps, 3))
    ts = np.zeros((nsteps, 1))
    rv_angle = np.zeros((nsteps, 1))
    for i in range(nsteps):
        norm_r = np.sqrt(rx**2 + ry**2 + rz**2)
        ax = -(rx *earth_mu ) /(norm_r**3)
        ay = -(ry * earth_mu) / (norm_r ** 3)
        az = -(rz * earth_mu) / (norm_r ** 3)

        r2 = norm_r**2
        z2 = rz**2

        # J2 correction
        r_J2 = 1.5 * J2 * ((earth_mu * earth_radius**2) / norm_r**4)
        ax_J2 = r_J2 * rx / norm_r * (5 * z2 / r2 - 1)
        ay_J2 = r_J2 * ry / norm_r * (5 * z2 / r2 - 1)
        az_J2 = r_J2 * rz / norm_r * (5 * z2 / r2 - 3)

        vx += (ax * dt) + (ax_J2 * dt)
        vy += (ay * dt) + (ay_J2 * dt)
        vz += (az * dt) + (az_J2 * dt)
        rx += vx * dt
        ry += vy * dt
        rz += vz * dt

        rs[i] = np.array((rx, ry, rz))
        vs[i] = np.array((vx, vy, vz))
        ts[i] = i*50
        rv_angle[i] = angle(rs[i], vs[i])

    hs = np.cross(rs, vs)

    plt.figure()
    plt.plot(ts/3600, np.linalg.norm(vs, axis=1))


    plt.figure()
    plt.plot(ts / 3600, np.linalg.norm(rs, axis=1))

    plt.figure()
    plt.plot(ts / 3600,rv_angle)


    plt.show()
    print()

