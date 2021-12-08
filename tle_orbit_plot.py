import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')


def ellipse(a, b):
    x = a * np.cos(np.linspace(-np.pi, 0, 250))
    x_return = x[len(x): 0: -1]

    y = np.sqrt(((1 - ((np.multiply(x, x)) / a ** 2)) * (b ** 2)))
    y_return = -np.sqrt(((1 - ((np.multiply(x_return, x_return)) / a ** 2)) * (b ** 2)))

    x = np.concatenate((x, x_return))
    y = np.concatenate((y, y_return))
    z = np.zeros(len(x))
    return x, y, z

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



if __name__ == '__main__':
    # ellipse array
    earth_radius = 6378.0  # km
    apogee = 6801.690554265845
    perigee = 6795.656051842055
    a = 6798.67330305395
    c = a - perigee
    b = np.sqrt((a**2) - (c**2))
    x, y, z = ellipse(a, b)
    x = x - c
    pos = np.vstack((x, y, z))
    s = 260.0308
    t = 51.6433
    p = 278.8265

    rotated = rotation313(pos, s, t, p)




    # plotting stuff
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plt.plot(x, y, z, 'r-', label='ISS', zorder=2, alpha=0.4)
    plt.plot(rotated[0], rotated[1],rotated[2], 'c-', label='rotated')
    plt.legend()
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    _x = earth_radius * np.cos(_u) * np.sin(_v)
    _y = earth_radius * np.sin(_u) * np.sin(_v)
    _z = earth_radius * np.cos(_v)
    ax.plot_surface(_x,_y,_z, color='b', label='Earth', zorder=1, alpha=0.25)

    ax.set_xlabel(['X (km)'])
    ax.set_ylabel(['Y (km)'])
    ax.set_zlabel(['Z (km)'])

    plt.show()
