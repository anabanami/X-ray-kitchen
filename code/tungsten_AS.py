import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, um
import scipy.constants as const
import xri
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

plt.rcParams['figure.dpi'] = 200

def thicc(x, y, R):
    # # Create cylindrical object projected thickness
    T = np.zeros_like(x * y)
    T[0:20,0:100] = 1
    T = 2 * np.sqrt(R ** 2 - x ** 2)
    T = np.nan_to_num(T)
    T = gaussian_filter(T, sigma=4)
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
    plt.plot(I[-1])
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profile")
    plt.show()


def globals():
    # constants
    h = const.h # 6.62607004e-34 * J * s
    c = const.c # 299792458 * m / s

    # # Discretisation parameters
    # # # x-array parameters
    # n = 1024
    # n_x = n
    # x_max = (n_x / 2) * 5 * um
    # x = np.linspace(-x_max, x_max, n_x, endpoint=True)
    # delta_x = x[1] - x[0]
    # # # y-array parameters
    # n_y = n
    # y_max = (n_y / 2) * 5 * um
    # y = np.linspace(-y_max, y_max, n_y, endpoint=True)
    # delta_y = y[1] - y[0]
    # y = y.reshape(n_y, 1)

    # # Matching LAB
    # Magnification
    # M = 1
    # M = 2.5
    M = 4.0

    # # x-array parameters
    delta_x = 55 * um / M
    x_max = 35 * mm / M
    x_min = -x_max
    n_x = int((x_max - x_min) / delta_x)
    print(f"\n{n_x = }")
    x = np.linspace(-x_max, x_max, n_x, endpoint=True) 
    # # y-array parameters
    delta_y = 55 * um / M
    y_max = 7 * mm / M
    y_min = -y_max
    n_y = int((y_max - y_min) / delta_y)
    print(f"\n{n_y = }")
    y = np.linspace(-y_max, y_max, n_y, endpoint=True).reshape(n_y, 1)

    ## Parameters from X-ray attenuation calculator   
    ### TUNGSTEN PEAKS ###
    # E = 8.1 # keV # W
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber1
    # # Material = water, density = 1 g/cm**3
    # δ = 3.52955e-06
    # μ = 999.13349 # per m
    # β = μ / (2 * k)

    E = 9.7 # keV # W 
    λ = h * c / (E * 1000 * const.eV)
    k = 2 * np.pi / λ # x-rays wavenumber2
    # Material = water, density = 1 g/cm**3
    δ =  2.45782e-06
    μ = 583.22302 # per m
    β = μ / (2 * k)

    # E = 11.2 # keV # W
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber3
    # # Material = water, density = 1 g/cm**3
    # δ = 1.84196e-06
    # μ = 381.85592 # per m
    # β = μ / (2 * k)

    # # Bremsstrahlung radiation peak - W @ 35kV
    # E = 21 # keV 
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ = 5.22809e-07
    # μ = 72.52674 # per m
    # β = μ / (2 * k)


    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(n_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(n_y, delta_y).reshape(n_y, 1)

    # Cylinder parameters
    D = 4 * mm
    R = D / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 10 * mm


    return M, x, y, n_x, n_y, delta_x, delta_y, E, k, δ, μ, β, kx, ky, R, z_c, x_c, height 


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    M, x, y, n_x, n_y, delta_x, delta_y, E, k, δ, μ, β, kx, ky, R, z_c, x_c, height  = globals()

    # z_eff = 1

    z_final = 1 * m
    z_eff = (z_final - (z_final / M)) / M # eff propagation distance

    T = thicc(x, y, R)
    δT = δ * T
    βT = β * T

    print("Propagating Wavefield")
    I = xri.sim.propAS(δT, βT, E, z_eff, delta_x, supersample=3)
    np.save(f'1m_I1_2_M=4.npy', I)


    # plt.plot(I[-10,:])
    # plt.xlabel("x")
    # plt.ylabel("I(x)")
    # plt.title("Intensity profile")
    # plt.show()
