import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, um
import scipy.constants as const
import xri
from scipy.ndimage import gaussian_filter

plt.rcParams['figure.dpi'] = 150

# functions

def thicc(x, y, R):
    # # Create cylindrical object projected thickness
    T = np.zeros_like(x * y)
    T[0:20,0:100] = 1
    T = 2 * np.sqrt(R ** 2 - x ** 2)
    T = np.nan_to_num(T)
    T = gaussian_filter(T, sigma=2)
    ones_y=np.ones_like(y)
    # # Expand 1D to 2D with outer product
    T = np.outer(ones_y, T)
    # im = plt.imshow(T)
    return T

def plots_I(I):

    plt.imshow(I, origin='lower')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("I(x, y)")
    plt.show()

    # # PLOT I vs x (a single slice)
    plt.plot(x, I[900])
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profile")
    plt.show()


def globals():
    # constants
    h = const.h # 6.62607004e-34 * J * s
    c = const.c # 299792458 * m / s

    # # # Discretisation parameters
    # # x-array parameters
    n = 1024
    n_x = n
    x_max = (n_x / 2) * 5 * um
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    delta_x = x[1] - x[0]
    # # y-array parameters

    n_y = n
    y_max = (n_y / 2) * 5 * um
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)#.reshape(n_y, 1)
    delta_y = y[1] - y[0]
    y = y.reshape(n_y, 1)

    # Parameters from X-ray attenuation calculator   
    E1 = 22.1629 # keV # Ag k-alpha1 
    λ = h * c / (E1 * 1000 * const.eV)
    k1 = 2 * np.pi / λ  # x-rays wavenumber

    # # Material = water, density = 1 g/cm**3
    # δ1 = 469.337 * nm
    # μ1 = 64.55083 # per m
    # β1 = μ1 / (2 * k1)
    # # Material = ice, density = 0.92 g/cm**3
    # δ2 = 431.790 * nm
    # μ2 = 59.38677 # per m
    # β2 = μ2 / (2 * k1)


    # # Brain parameters from X-ray attenuation calculator
    # # Material = Brain, density = 1.03 g/cm**3
    # δ1 = 482.584 * nm
    # μ1 = 68.29402 # per m
    # β1 = μ1 / (2 * k1)

    # # Material = gray matter, density = ? g/cm**3
    # δ1 = 16.1 * nm # ?
    # μ1 = 68 # per m
    # β1 = μ1 / (2 * k1)
    # Material = white matter, density = ? g/cm**3
    δ2 = 16.1 * nm
    μ2 = 66 # per m
    β2 = μ2 / (2 * k1)
    # # Material = other, density = ? g/cm**3
    # δ3 = -2.837
    # μ3 =  ?? # per cm
    # β3 = μ3 / (2 * k1)


    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(n_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(n_y, delta_y).reshape(n_y, 1)

    # Cylinder1 parameters
    D1 = 4 * mm
    R1 = D1 / 2

    # Cylinder2 parameters
    D2 = 2 * mm
    R2 = D2 / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 10 * mm


    return x, y, n_x, n_y, delta_x, delta_y, E1, k1, kx, ky, R1, R2, z_c, x_c, δ1, μ1, β1, δ2, μ2, β2, height 


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, y, n_x, n_y, delta_x, delta_y, E1, k1, kx, ky, R1, R2, z_c, x_c, δ1, μ1, β1, δ2, μ2, β2, height = globals()

    z_final = 1 * m

    T1 = thicc(x, y, R1)
    T2 = thicc(x, y, R2)
    δT1 = δ1 * T1
    βT1 = β1 * T1
    δT2 = δ2 * T2
    βT2 = β2 * T2
    two_cylinders_δT = δT1 + (δ2 - δ1) * T2
    two_cylinders_βT = βT1 + (β2 - β1) * T2  

    # I = xri.sim.propAS(δT1, βT1, E1, z_final, delta_x, supersample=3)
    # I = xri.sim.propAS(δT2, βT2, E1, z_final, delta_x, supersample=3)
    I = xri.sim.propAS(two_cylinders_δT, two_cylinders_βT, E1, z_final, delta_x, supersample=3)
    plots_I(I)
