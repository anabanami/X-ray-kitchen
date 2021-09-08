import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, J, kg, s
from scipy import integrate

plt.rcParams['figure.dpi'] = 150

# functions

def y_sigmoid(y):
    # smoothing out the edges of the cylinder in the y-direction
    𝜎_y = 0.004 * mm
    S = np.abs(1 / (1 + np.exp(-(y - height/2) / 𝜎_y)) - 
        (1 / (1 + np.exp(-(y + height/2) / 𝜎_y))))
    return S # np.shape = (n_y, 1)


def δ(x, y, z, δ1):#, δ2):
    '''Refractive index: δ1 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    δ_array = (δ1 * (1 / (1 + np.exp((r - R) / 𝜎_x))) )#+ (δ2 - δ1) * (1 / (1 + np.exp((r - R) / 𝜎_x)))) * y_sigmoid(y)
    return δ_array # np.shape(δ_array) = (n_y, n_x)


def μ(x, y, z, μ1):#, μ2):
    '''attenuation coefficient: μ1 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    μ_array = (μ1 * (1 / (1 + np.exp((r - R) / 𝜎_x))) )# + (μ2 - μ1) * (1 / (1 + np.exp((r - R) / 𝜎_x)))) * y_sigmoid(y)
    return μ_array # np.shape(μ_array) = (n_y, n_x)


def phase(x, y):
    # phase gain as a function of the cylinder's refractive index
    Φ = np.zeros_like(x * y)
    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            Φ[i, j] = -k0 * integrate.quad(lambda z: δ(x_val, y_val, z, δ1),-2 * R, 2 * R)[0] #, δ2)
            print(f"{Φ[i, j]}")
    return Φ # np.shape(Φ) = (n_y, n_x)


def BLL(x, y):
    # TIE IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
    F = np.zeros_like(x * y)
    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            F[i, j] = integrate.quad(lambda z: μ(x_val, y_val, z, μ1),-2 * R, 2 * R)[0] #, μ2)
            print(f"{F[i, j]}")
    I = np.exp(- F) * I_initial
    return I # np.shape(I) = (n_y, n_x)


def Angular_spectrum(z, field):
    """Wave propagation using the angular spectrum method.
    Code follows Als-Nielsen, Elements of Modern X-ray Physics, p.324"""
    D_operator = np.exp(1j * k0 * z) * np.exp(-1j * (kx**2 * z + ky**2 * z) / (2 * k0))
    FT_field = fft2(field)
    return ifft2(D_operator * FT_field)


def plots_I(I):
    # # PLOT Phase contrast I in x, y
    # plt.imshow(I, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("I")
    # plt.show()

    # # PLOT I vs x (a single slice)
    plt.plot(x, I[np.int(n_y / 2),:])
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

    n_y = 20 # n
    y_max = 10 * mm
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)#.reshape(n_y, 1)
    delta_y = y[1] - y[0]
    y = y.reshape(n_y, 1)

    # # X-ray beam parameters
    # # (Beltran et al. 2010)
    E = 3.845e-15 * J 
    λ = h * c / E
    # # refraction and attenuation coefficients
    δ1 = 462.8 * nm # PMMA
    μ1 = 41.2 # per meter # PMMA
    # δ2 = 939.6 * nm # Aluminium
    # μ2 = 502.6 # per meter # Aluminium

    # wave number
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(n_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(n_y, delta_y).reshape(n_y, 1)

    # Blurring 
    𝜎_x = 0.01 * mm

    # Cylinder1 parameters
    D = 12.75 * mm
    R = D / 2

    # Cylinder2 parameters
    D2 = 6 * mm
    R2 = D2 / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 20 * mm # change to 10mm?


    return x, y, n_x, n_y, delta_x, delta_y, k0, kx, ky, R, R2, z_c, x_c, δ1, μ1, 𝜎_x, height # add δ2, μ2, 


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, y, n_x, n_y, delta_x, delta_y, k0, kx, ky, R, R2, z_c, x_c, δ1, μ1, 𝜎_x, height = globals() # add δ2, μ2,

    # # ICS
    I_initial = np.ones_like(x * y)
    I_0 = BLL(x, y)
    Φ = phase(x, y)

    Ψ_0 = np.sqrt(I_0) * np.exp(1j*Φ)

    z_final = 20 * cm
    Ψ = Angular_spectrum(z_final, Ψ_0)
    I = np.abs(Ψ**2)

    plots_I(I)
