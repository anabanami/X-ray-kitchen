import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, J, kg, s, keV
from scipy import integrate
import xri
from scipy.ndimage import gaussian_filter

plt.rcParams['figure.dpi'] = 150

# functions

def thicc(x, y):
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


def phase(x, y, δ):
    # phase gain as a function of the cylinder's refractive index
    Φ = - k0 * δ * thicc(x, y)
    return Φ # np.shape(Φ) = (n_y, n_x)

def BLL(x, y, μ):
    # IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
    I = np.exp(- μ * thicc(x, y)) * I_initial
    return I # np.shape(I) = (n_y, n_x)


def Angular_spectrum(z, field):
    """Wave propagation using the angular spectrum method.
    Code follows Als-Nielsen, Elements of Modern X-ray Physics, p.324"""
    D_operator = np.exp(1j * k0 * z) * np.exp(-1j * (kx**2 * z + ky**2 * z) / (2 * k0))
    FT_field = fft2(field)
    return ifft2(D_operator * FT_field)


def plots_I(I):
    # PLOT Phase contrast I in x, y
    # plt.imshow(I, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("I")
    # plt.show()

    # # PLOT I vs x (a single slice)
    plt.plot(I)
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profile")
    plt.show()



def globals():
    # constants
    h = 6.62607004e-34 * m**2 * kg / s
    c = 299792458 * m / s

    # # # Discretisation parameters
    # # x-array parameters
    n = 1024
    n_x = n
    x_max = 10 * mm
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    delta_x = x[1] - x[0]
    # # y-array parameters

    n_y = n
    y_max = 10 * mm
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)#.reshape(n_y, 1)
    delta_y = y[1] - y[0]
    y = y.reshape(n_y, 1)

    # # # X-ray beam parameters
    # # # (Beltran et al. 2010)
    # E = 3.845e-15 * J 
    # λ = h * c / E
    # # # refraction and attenuation coefficients
    # δ1 = 462.8 * nm # PMMA
    # μ1 = 41.2 # per meter # PMMA
    # # δ2 = 939.6 * nm # Aluminium
    # # μ2 = 502.6 # per meter # Aluminium

    # # Parameters from Energy_Dispersion_Sim-1.py
    # # Material = water
    E1 = 22.1629 * keV # Ag k-alpha1
    δ1 = 4.68141e-07
    μ1 = 64.38436
    λ = h * c / E1
    # E2 = 24.942 #keV - Ag k-beta1
    # δ2 = 3.69763e-07
    # μ2 = 50.9387
    # λ2 = h * c / E2

    # wave number
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(n_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(n_y, delta_y).reshape(n_y, 1)

    # Cylinder1 parameters
    D = 4 * mm
    R = D / 2

    # Cylinder2 parameters
    D2 = 6 * mm
    R2 = D2 / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 20 * mm # change to 10mm?


    return x, y, n_x, n_y, delta_x, delta_y, E1, k0, kx, ky, R, z_c, x_c, δ1, μ1, height # add δ2, μ2, 


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, y, n_x, n_y, delta_x, delta_y, E1, k0, kx, ky, R, z_c, x_c, δ1, μ1, height = globals() # add δ2, μ2,

    # # ICS
    I_initial = np.ones_like(x * y)
    I_0 = BLL(x, y, μ1)
    Φ = phase(x, y, δ1)
    # Ψ_0 = np.sqrt(I_0) * np.exp(1j*Φ)

    # z_final = 1 * m
    # Ψ = Angular_spectrum(z_final, Ψ_0)
    # I = np.abs(Ψ ** 2)

    print(f"{x[-1] = }")
    print(f"{y[-1] = }")




    z_final = 1 * m
    beta1 = μ1 / (2 * k0)
    T = thicc(x, y)
    Prop1 = xri.sim.propAS(δ1*T, beta1*T, E1, z_final, delta_x, supersample=3)
    I = Prop1[900]
    plots_I(I)
