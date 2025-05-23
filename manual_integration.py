import numpy as np
import pickle

earth_mu = 398600.4418
earth_radius = 6378.135

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

    for i in range(100):
        norm_r = np.sqrt(rx**2 + ry**2 + rz**2)
        ax = -(rx *earth_mu ) /(norm_r**3)
        ay = -(ry * earth_mu) / (norm_r ** 3)
        az = -(rz * earth_mu) / (norm_r ** 3)

        



    print()
    # rx, ry, rz, vx, vy, vz = y
    # r = np.array([rx, ry, rz])
    # v = np.array([vx, vy, vz])
    # # norm of radius vector
    # norm_r = np.linalg.norm(r)
    #
    # r2 = norm_ r* *2
    # z2 = r[2 ]* *2
    # # Earth's gravitational acceleration
    # ax, ay, az = -( r *earth_mu ) /(norm_ r* *3)
    #
    # _a = np.linalg.norm(np.array((ax, ay, az)))
    #
    # # J2 correction
    # r_J2 = 1.5 * J2 * ((earth_mu * earth_radiu s* *2) / norm_ r* *4)
    # ax_J2 = r_J2 * r[0] / norm_r * (5 * z2 / r2 - 1)
    # ay_J2 = r_J2 * r[1] / norm_r * (5 * z2 / r2 - 1)
    # az_J2 = r_J2 * r[2] / norm_r * (5 * z2 / r2 - 3)
    #
    # _aj2 = np.linalg.norm(np.array((ax_J2, ay_J2, az_J2)))
    # # atmospheric drag
    # # drag_x = air_density_ratio(norm_r) * bstar / earth_radius * (vx ** 2)
    # # drag_y = air_density_ratio(norm_r) * bstar / earth_radius * (vy ** 2)
    # # drag_z = air_density_ratio(norm_r) * bstar / earth_radius * (vz ** 2)
    #
    # ax += ax_J2 # + drag_x
    # ay += ay_J2 # + drag_y
    # az += az_J2 # + drag_z
