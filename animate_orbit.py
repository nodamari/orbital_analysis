from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import pickle

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
earth_radius = 6378.0 #n
# plotting Earth
_u, _v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
_x = earth_radius * np.cos(_u) * np.sin(_v)
_y = earth_radius * np.sin(_u) * np.sin(_v)
_z = earth_radius * np.cos(_v)

# axes label
ax.set_xlabel(['X (km)'])
ax.set_ylabel(['Y (km)'])
ax.set_zlabel(['Z (km)'])
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.85, 0.85, 1, 1]))
plt.legend()
ax.plot_surface(_x, _y, _z, color='honeydew', label='Earth', zorder=1, alpha=0.3)

with open("states.pkl", "rb") as f:
    states = pickle.load(f)

rs = states["rs"]
vs = states["vs"]
e = states["e"].T

h = np.cross(rs, vs)
vh_cross = np.cross(vs, h)
h = h.T
rs = rs.T
vs = vs.T
vh_cross = vh_cross.T

N = 150


line, = ax.plot(rs[0:1, 0], rs[0:1, 1], rs[0:1, 2])

# ecc = ax.quiver(0, 0, 0,
#           e[0:1, 0] * 7000000,
#           e[0:1, 1] * 7000000,
#           e[0:1, 2] * 7000000, color='orange')

pos_vec = ax.quiver(0, 0, 0,
          rs[0:1, 0] ,
          rs[0:1, 1] ,
          rs[0:1, 2] , color='orange')

vel_vec = ax.quiver(rs[0:1, 0], rs[0:1, 1], rs[0:1, 2],
          vs[0:1, 0] * 300,
          vs[0:1, 1] * 300,
          vs[0:1, 2] * 300, color='deepskyblue')


h_vec = ax.quiver(0, 0, 0,
          h[0:1, 0]/7 ,
          h[0:1, 1]/7 ,
          h[0:1, 2] /7, color='green')


vh_cross_vec = ax.quiver(0, 0, 0,
          vh_cross[0:1, 0]/100 ,
          vh_cross[0:1, 1]/100 ,
          vh_cross[0:1, 2] /100, color='hotpink')

def update(num):

    line.set_data(rs[:2, :num])
    line.set_3d_properties(rs[2, :num])

    # global ecc
    # ecc.remove()
    # ecc = ax.quiver(0, 0, 0, e[0, num]*7000000, e[1, num]*7000000, e[2, num]*7000000, color="orange")


    global pos_vec
    pos_vec.remove()
    pos_vec = ax.quiver(0, 0, 0,
                        rs[0, num],
                        rs[1, num],
                        rs[2, num], color='orange')

    # global vel_vec
    # vel_vec.remove()
    # vel_vec = ax.quiver(rs[0, num], rs[1, num], rs[2, num], vs[0, num]*300, vs[1, num]*300, vs[2, num]*300, color="deepskyblue")

    global h_vec
    h_vec.remove()
    h_vec = ax.quiver(0, 0, 0,
                        h[0, num]/7,
                        h[1, num]/7,
                        h[2, num]/7, color='green')


    global vh_cross_vec
    vh_cross_vec.remove()
    vh_cross_vec = ax.quiver(0, 0, 0,
                        vh_cross[0, num]/100,
                        vh_cross[1, num]/100,
                        vh_cross[2, num]/100, color='hotpink')


# Setting the axes properties
ax.set_xlim3d([-7000, 7000])
ax.set_xlabel('X')

ax.set_ylim3d([-7000, 7000])
ax.set_ylabel('Y')

ax.set_zlim3d([-7000, 7000])
ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, N,  interval=100, blit=False)
#ani = animation.FuncAnimation(fig, update, N, fargs=(e, ecc), interval=1, blit=False)
ani.save('matplot003.gif', writer='imagemagick', fps=50)
plt.show()