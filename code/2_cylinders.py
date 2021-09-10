import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, um, J, kg, s, eV
import scipy.constants as const
import xri
from scipy.ndimage import gaussian_filter

plt.rcParams['figure.dpi'] = 150

# functions

def thicc(x, y):
    # # Create cylindrical object projected thickness
    T = np.zeros_like(x * y)
    T[0:20,0:100] = 1
    T = 2 * np.sqrt(R1 ** 2 - x ** 2) #+ 2 * np.sqrt(R2 ** 2 - x ** 2)
    T = np.nan_to_num(T)
    T = gaussian_filter(T, sigma=2)
    ones_y=np.ones_like(y)
    # # Expand 1D to 2D with outer product
    T = np.outer(ones_y, T)
    # im = plt.imshow(T)
    # plt.colorbar()
    # plt.show()
    # ass
    return T


# def phase(x, y, δ):
#     # phase gain as a function of the cylinder's refractive index
#     Φ = - k0 * δ * thicc(x, y)
#     return Φ # np.shape(Φ) = (n_y, n_x)


# def BLL(x, y, μ):
#     # IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
#     I = np.exp(- μ * thicc(x, y)) * I_initial
#     return I # np.shape(I) = (n_y, n_x)


def plot_I(I):
    # # PLOT I vs x (a single slice)
    plt.plot(I)
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

    # # X-ray beam parameters
    # # (Beltran et al. 2010)
    E0 = 24 # keV 
    λ = h * c / E0
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # # refraction and attenuation coefficients
    δ1 = 462.8 * nm # PMMA
    μ1 = 41.2 # per meter # PMMA
    β1 = μ1 / (2 * k0)
    δ2 = 939.6 * nm # Aluminium
    μ2 = 502.6 # per meter # Aluminium
    β2 = μ2 / (2 * k0)

    # # Parameters from Energy_Dispersion_Sim-1.py
    # # Material = water
    # E1 = 22.1629 # keV # Ag k-alpha1 
    # λ = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ  # x-rays wavenumber
    # δ1 = 468.141 * nm
    # μ1 = 64.38436 # per meter
    # β1 = μ1 / (2 * k1)

    # E2 = 24.942 # keV - Ag k-beta1
    # λ2 = h * c / E2
    # k2 = 2 * np.pi / λ2  # x-rays wavenumber
    # δ2 = 3.69763e-07
    # μ2 = 50.9387

    # β2 = μ2 / (2 * k0)

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
    height = 10 * mm # change to 10mm?


    return x, y, n_x, n_y, delta_x, delta_y, E0, k0, kx, ky, R1, R2, z_c, x_c, δ1, μ1, β1, δ2, μ2, β2,  height


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, y, n_x, n_y, delta_x, delta_y, E0, k0, kx, ky, R1, R2, z_c, x_c, δ1, μ1, β1, δ2, μ2, β2,  height = globals()

    z_final = 1 * m

    T = thicc(x, y)

    δ1T = δ1 * T
    β1T = β1 * T

    Prop1 = xri.sim.propAS(δ1T, β1T, E0, z_final, delta_x, supersample=3)
    # Prop2 = xri.sim.propAS(δ2*T, β2*T, E0, z_final, delta_x, supersample=3)

    # # # PLOT Phase contrast I in x, y
    # plt.imshow(Prop1, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("I(x, y)")
    # plt.show()

    I = Prop1[900]
    print(f"{I = }")
    plot_I(I)
