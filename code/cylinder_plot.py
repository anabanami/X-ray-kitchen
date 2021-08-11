import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

plt.rcParams['figure.dpi'] = 200

def plot_3D_cylinder(radius, height, elevation=0, resolution=512, color='salmon', x_center=0, y_center=0):

    fig = plt.figure()
    ax = Axes3D(fig)
 
    ax.view_init(elev=135, azim=270)

    z = np.linspace(elevation, elevation + height, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    𝚹, Z = np.meshgrid(theta, z)

    X = radius * np.cos(𝚹) + x_center
    Y = radius * np.sin(𝚹) + y_center

    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2 * y_center - Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation + height, zdir="z")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


# parameters
radius = 3
height = 12
elevation = -6
resolution = 1024
colour = 'orange'
x_center = 3
y_center = 3

plot_3D_cylinder(
    radius,
    height,
    elevation=elevation,
    resolution=resolution,
    color=colour,
    x_center=x_center,
    y_center=y_center,
)
