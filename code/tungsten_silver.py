import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, um
import scipy.constants as const
import xri
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

plt.rcParams['figure.dpi'] = 150
# folder = Path('Simulations')
# os.makedirs(folder, exist_ok=True)
# os.system(f'rm {folder}/*.png')
# functions

def thicc(x, y, R):
    # # Create cylindrical object projected thickness
    T = np.zeros_like(x * y)
    T[0:20,0:100] = 1
    T = 2 * np.sqrt(R ** 2 - x ** 2)
    T = np.nan_to_num(T)
    T = gaussian_filter(T, sigma=3)
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

    # Discretisation parameters
    # # x-array parameters
    n = 1024
    n_x = n
    x_max = (n_x / 2) * 5 * um
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    delta_x = x[1] - x[0]
    # # y-array parameters
    n_y = n
    y_max = (n_y / 2) * 5 * um
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)
    delta_y = y[1] - y[0]
    y = y.reshape(n_y, 1)

    # # # <---- TEST
    # E1 = 22.1629 # keV # Ag k-alpha1 
    # λ = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ  # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ1 = 469.337 * nm
    # μ1 = 64.55083 # per m
    # β1 = μ1 / (2 * k1)

    ### TUNGSTEN ###

    # # Parameters from X-ray attenuation calculator   
    # E1 = 8.1 # keV # W
    # λ1 = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ1 # x-rays wavenumber1
    # # Material = water, density = 1 g/cm**3
    # δ1 = 352.955 * nm
    # μ1 = 999.13349 # per m
    # β1 = μ1 / (2 * k1)

    E1 = 19 # keV # W 
    λ1 = h * c / (E1 * 1000 * const.eV)
    k1 = 2 * np.pi / λ1 # x-rays wavenumber2
    # Material = water, density = 1 g/cm**3
    δ1 =  6.38803e-07 
    μ1 =  91.32598 # per m
    β1 = μ1 / (1 * k1)


    # E1 = 9.7 # keV # W 
    # λ1 = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ1 # x-rays wavenumber2
    # # Material = water, density = 1 g/cm**3
    # δ1 =  245.782 * nm
    # μ1 = 583.22302 # per m
    # β1 = μ1 / (1 * k1)

    # E1 = 11.2 # keV # W
    # λ1 = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ1 # x-rays wavenumber3
    # # Material = water, density = 1 g/cm**3
    # δ1 = 184.196 * nm
    # μ1 = 381.85592 # per m
    # β1 = μ1 / (2 * k1)

    # ## SILVER ##
    # E1 = 21.99 # keV
    # λ = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ  # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ1 = 476.753 * nm
    # μ1 = 65.62992 # per m
    # β1 = μ1 / (2 * k1)

    # E1 = 24.911 # keV
    # λ = h * c / (E1 * 1000 * const.eV)
    # k1 = 2 * np.pi / λ  # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ1 = 371.427 * nm
    # μ1 = 51.15927 # per m
    # β1 = μ1 / (2 * k1)


    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(n_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(n_y, delta_y).reshape(n_y, 1)

    # Cylinder1 parameters
    D1 = 4 * mm
    R1 = D1 / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 10 * mm


    return x, y, n_x, n_y, delta_x, delta_y, E1, k1, kx, ky, R1, z_c, x_c, δ1, μ1, β1, height 


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    # x, y, n_x, n_y, delta_x, delta_y, E1, k1, kx, ky, R1, z_c, x_c, δ1, μ1, β1, height = globals()

    # z_final =  2.5 * m # propagation distance

    # T = thicc(x, y, R1)
    # δT1 = δ1 * T
    # βT1 = β1 * T

    # print("Propagating Wavefield")
    # I = xri.sim.propAS(δT1, βT1, E1, z_final, delta_x, supersample=3)
    # np.save(f'I1_6.npy', I)

    # # # Re-bin step each pixel should now be 20um (for the case of 5um pixels)
    # I = zoom(I, 0.25, order=3)
    # plots_I(I)

    I1 = np.load("I1_1.npy")
    I2 = np.load("I1_2.npy")
    I3 = np.load("I1_3.npy")
    I6 = np.load("I1_6.npy")

    I1 = zoom(I1, 0.25, order=3)
    I2 = zoom(I2, 0.25, order=3)
    I3 = zoom(I3, 0.25, order=3)
    I6 = zoom(I6, 0.25, order=3)

    plt.plot(I1[-1], label="E = 8.1 keV")
    plt.plot(I2[-1], label="E = 9.7 keV")
    plt.plot(I3[-1], label="E = 11.2 keV")
    plt.plot(I6[-1], label="E = 19 keV")
    plt.axvline(27, color="grey", ls=":")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profiles for tungsten peaks")
    plt.show()

    I4 = np.load("I1_4.npy")
    I5 = np.load("I1_5.npy")

    I4 = zoom(I4, 0.25, order=3)
    I5 = zoom(I5, 0.25, order=3)

    plt.plot(I4[-1], label="E = 21.99 keV")
    plt.plot(I5[-1], label="E = 24.911 keV")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profiles for silver peaks")
    plt.show()




