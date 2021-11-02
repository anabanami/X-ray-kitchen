import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft, fft2, ifft2
from physunits import m, cm, mm, nm, J, kg, s, keV

# plt.rcParams['figure.dpi'] = 150

# functions

def y_sigmoid(y):
    𝜎 = 0.004 * mm
    S = np.abs(1 / (1 + np.exp(-(y - height/2) / 𝜎)) - (1 / (1 + np.exp(-(y + height/2) / 𝜎))))
    return S # np.shape = (n_y, 1)


def δ(x, y, z):
    '''Refractive index: δ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    𝜎 = 0.01 * mm
    δ_array = δ0 * (1 / (1 + np.exp((r - R) / 𝜎))) * y_sigmoid(y)
    return δ_array # np.shape(δ_array) = (n_y, n_x)


def μ(x, y, z):
    '''attenuation coefficient: μ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    𝜎 = 0.01 * mm
    μ_array = μ0 * (1 / (1 + np.exp((r - R) / 𝜎))) * y_sigmoid(y)
    return μ_array # np.shape(μ_array) = (n_y, n_x)


def phase(x, y):
    # phase gain as a function of the cylinder's refractive index
    z = np.linspace(-2 * R, 2 * R, 65536, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    Φ = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        Φ += -k0 * δ(x, y, z_value) * dz
    return Φ # np.shape(Φ) = (n_y, n_x)


def BLL(x, y):
    # TIE IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
    z = np.linspace(-2 * R, 2 * R, 65536, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    F = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        F += μ(x, y, z_value) * dz
    I = np.exp(- F) * I_initial
    return I # np.shape(I) = (n_y, n_x)


def gradΦ_laplacianΦ(Φ):
    FT2D_Φ = fft2(Φ)
    dΦ_dx = ifft2(1j * kx * FT2D_Φ)
    dΦ_dy = ifft2(1j * ky * FT2D_Φ)
    lap_Φ = ifft2(-(kx ** 2 + ky ** 2) * FT2D_Φ)
    return dΦ_dx, dΦ_dy, lap_Φ


def TIE(z, I):
    '''The intensity and phase evolution of a paraxial monochromatic
    scalar electromagnetic wave on propagation (2D)'''
    FT2D_I = fft2(I)
    dI_dz = (-1 / k0) * (
        np.real(
            ifft2(1j * kx * FT2D_I) * dΦ_dx
            + ifft2(1j * ky * FT2D_I) * dΦ_dy
            + I * lap_Φ
        )
    )
    return dI_dz  # np.shape(dI_dz) = (n_y, n_x)


def Runge_Kutta(z, delta_z, I):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    k1 = TIE(z, I)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2)
    k4 = TIE(z + delta_z, I + k3 * delta_z)    
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # shape = (n_y, n_x)


def finite_diff(z, I):
    # first order finite differences
    I_z = I + z * TIE(z, I)
    return I_z


def propagation_loop(I_0):
    # RK Propagation loop parameters
    i = 0
    z = 0
    z_final = 1000 * mm
    delta_z = 1 * mm  # (n_z = 100)

    I = I_0

    I_list = []
    while z < z_final:

        print(f"{i = }")

        # spatial evolution step
        I = Runge_Kutta(z, delta_z, I)
        if not i % 10:
            I_list.append(I)
        i += 1
        z += delta_z

    I_list = np.array(I_list)
    print(f"{np.shape(I_list) = }") #  np.shape(I_list) = (n_z / 10, n_x)

    np.save(f'I_list.npy', I_list)
    return I_list

def globals():

    # constants
    h = 6.62607004e-34 * m**2 * kg / s
    c = 299792458 * m / s

    # x-array parameters
    n_all = 512

    n_x = n_all * 2
    x_max = 10 * mm
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    delta_x = x[1] - x[0]
    size_x = x.size

    # y-array parameters
    n_y = n_all
    y_max = 10 * mm
    y = np.linspace(-y_max, y_max, n_y, endpoint=False).reshape(n_y, 1)
    delta_y = y[1] - y[0]
    size_y = y.size

    # # # refraction and attenuation coefficients (Beltran et al. 2010)
    # δ0 = 4.628e-7
    # μ0 = 41.2 # per meter 
    # # # X-ray beam parameters
    # E = 3.845e-15 * J 
    # λ = h * c / E
    # k0 = 2 * np.pi / λ  # x-rays wavenumber

    #Parameters as per energy_dispersion_Sim-1.py
    energy1 = 22.1629 * keV #- Ag k-alpha1
    δ0 = 4.68141e-7
    μ0 = 64.38436 
    λ = h * c / energy1

    # energy2 = 3.996e-15  * J # = 24.942 * keV # - Ag k-beta1
    # δ0 = 369.763 *nm
    # μ0 = 50.9387 
    # λ = h * c / energy2

    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # Cylinder parameters
    D = 12.75 * mm
    R = D / 2
    z_c = 0 * mm
    x_c = 0 * mm
    height = 20 * mm

    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(size_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(size_y, delta_y).reshape(size_y, 1)

    return n_x, n_y, x, y, k0, R, z_c, x_c, height, δ0, μ0, kx, ky


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    n_x, n_y, x, y, k0, R, z_c, x_c, height, δ0, μ0, kx, ky = globals()

    # # ICS
    I_initial = np.ones_like(x * y)

    Φ = phase(x, y)
    # np.save(f'phase_x_y.npy', Φ)
    I_0 = BLL(x, y)
    # np.save(f'intensity_x_y.npy', I_0)

    # Φ derivatives 
    dΦ_dx, dΦ_dy, lap_Φ = gradΦ_laplacianΦ(Φ)
    # # Fourth order Runge-Kutta
    I_list = propagation_loop(I_0)

    ##################### PLOTS & TESTS #############################

    # # # Load file
    I_list = np.load("I_list.npy")  # np.shape(I_list) = (n_z / 10, n_y,  n_x)
    I = I_list[-1,:, :]
    # Φ = np.load("phase_x_y.npy")
    # I_0 = np.load("intensity_x_y.npy")

    # PLOT Phase contrast I in x, y
    plt.figure(figsize=(5, 4))
    plt.imshow(I[50:-50], origin='lower')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("I")
    plt.show()

    # PLOT I vs x (a single slice)
    plt.figure(figsize=(5, 4))
    plt.plot(x, I[np.int(n_y / 2),:])
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profile")
    plt.show()
