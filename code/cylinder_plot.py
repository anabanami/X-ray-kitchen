import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

def plot_3D_cylinder(radius, height, elevation=0, resolution=512, color='salmon', z_center=0, x_center=0):

    fig = plt.figure()
    ax = Axes3D(fig)

    y = np.linspace(elevation, elevation + height, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    𝚹, Y = np.meshgrid(theta, y)

    Z = radius * np.cos(𝚹) + z_center
    X = radius * np.sin(𝚹) + x_center

    ax.plot_surface(Y, Z, X, linewidth=0, color=color)

    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')

    plt.show()


# parameters
radius = 3
height = 12
elevation = -6
resolution = 1024
colour = 'orange'
z_center = 3
x_center = 3


plot_3D_cylinder(
    radius,
    height,
    elevation=elevation,
    resolution=resolution,
    color=colour,
    z_center=z_center,
    x_center=x_center
)

