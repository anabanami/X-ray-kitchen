import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft, fft2, ifft2
from physunits import m, cm, mm, nm

plt.rcParams['figure.dpi'] = 150

# functions

def y_sigmoid(y):
    𝜎 = 0.005 * mm
    S = np.abs(1 / (1 + np.exp(-(y - h/2) / 𝜎)) - (1 / (1 + np.exp(-(y + h/2) / 𝜎))))
    return S # np.shape = (n_y, 1)


def δ(x, y, z):
    '''Refractive index: δ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    δ0 = 462.8 * nm
    # δ_array = np.zeros_like(x * y)
    # δ_array[(z-z_c)**2 + (x - x_c)**2 <= R] = δ0
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    𝜎 = 0.005 * mm
    δ_array = δ0 * (1 / (1 + np.exp((r - R) / 𝜎))) * y_sigmoid(y)
    return δ_array # np.shape(δ_array) = (n_y, n_x)


def μ(x, y, z):
    '''attenuation coefficient: μ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    μ0 = 41.2 # per meter
    # μ_array = np.zeros_like(x * y)
    # μ_array[(z-z_c)**2 + (x - x_c)**2 <= R] = μ0
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    𝜎 = 0.1 * mm
    μ_array = μ0 * (1 / (1 + np.exp((r - R) / 𝜎))) * y_sigmoid(y)
    return μ_array # np.shape(μ_array) = (n_y, n_x)


def phase(x, y):
    # phase gain as a function of the cylinder's refractive index
    z = np.linspace(-2 * R, 2 * R, 1024, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    Φ = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        Φ += -k0 * δ(x, y, z_value) * dz
    return Φ # np.shape(Φ) = (n_y, n_x)


def BLL(x, y):
    # TIE IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
    z = np.linspace(-2 * R, 2 * R, 1024, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    F = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        F += μ(x, y, z_value) * dz
    I = np.exp(- F) * I_0
    return I # np.shape(I) = (n_y, n_x)


def TIE(z, I, Φ):
    '''The intensity and phase evolution of a paraxial monochromatic
    scalar electromagnetic wave on propagation (2D)'''
    FT2D_I = fft2(I)
    FT2D_Φ = fft2(Φ)
    dI_dz = (-1 / k0) * (
        np.real(
            ifft2(1j * kx * FT2D_I) * ifft2(1j * kx * FT2D_Φ)
            + ifft2(1j * ky * FT2D_I) * ifft2(1j * ky * FT2D_Φ)
            + I * ifft2(-(kx ** 2 + ky ** 2) * FT2D_Φ)
        )
    )
    return dI_dz  # np.shape(dI_dz) = (n_y, n_x)


def Runge_Kutta(z, delta_z, I, Φ):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    k1 = TIE(z, I, Φ)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2, Φ)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2, Φ)
    k4 = TIE(z + delta_z, I + k3 * delta_z, Φ)    
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # shape = (n_y, n_x)


def finite_diff(z, I):
    # first order finite differences
    I_z = I + z * TIE(z, I, Φ)
    plt.imshow(I_z, origin="lower")
    plt.show()
    return I_z


def globals():
    # x-array parameters
    n_all = 256

    n_x = n_all
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

    # X-ray beam parameters
    λ = 0.05166 * nm  # x-rays wavelength
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # Cylinder parameters
    D = 12.75 * mm
    R = D / 2
    z_c = 0 * mm
    x_c = 0 * mm
    h = 20 * mm

    # For Fourier space
    kx = 2 * np.pi * np.fft.fftfreq(size_x, delta_x)
    ky = 2 * np.pi * np.fft.fftfreq(size_y, delta_y).reshape(size_y, 1)

    return n_x, n_y, x, y, k0, R, z_c, x_c, h, kx, ky


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    n_x, n_y, x, y, k0, R, z_c, x_c, h, kx, ky = globals()

    #ICS
    I_0 = np.ones_like(x * y)
    Φ = phase(x, y)
    np.save(f'phase_x_y.npy', Φ)

    I = BLL(x, y)
    np.save(f'intensity_x_y.npy', I)

    # ########################## RK LOOP ###############################

    # Φ = np.load("phase_x_y.npy")
    # I_0 = np.load("intensity_x_y.npy")

    # RK Propagation loop parameters
    i = 0
    z = 0
    z_final = 1000 * mm
    delta_z = 0.1 * mm  # (n_z = 10000)

    I_list = []
    while z < z_final:

        print(f"{i = }")

        # spatial evolution step
        I = Runge_Kutta(z, delta_z, I_0, Φ)
        if not i % 10:
            I_list.append(I)
        i += 1
        z += delta_z

    I_list = np.array(I_list)
    print(f"{np.shape(I_list) = }") #  np.shape(I_list) = (n_z / 10, n_x)

    np.save(f'I_list.npy', I_list)


    ###################### PLOTS & TESTS #############################
    # # # Finite Difference 1st order
    # # finite_diff(z_final, I)

    # # Load file
    # I_list = np.load("I_list.npy")  # np.shape(I_list) = (n_z / 10, n_y,  n_x)
    # I = I_list[-1,:, :]
    # print(f"{np.shape(I) = }")
    # Φ = np.load("phase_x_y.npy")
    # I_0 = np.load("intensity_x_y.npy")

    # # PLOT Phase contrast I in x, y
    # plt.imshow(I, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Absorption profile")
    # plt.show()

    # # PLOT Phase contrast I in x, y
    # plt.imshow(Φ, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Phase")
    # plt.show()

    # # PLOT I/I_0 in x, y
    # plt.imshow(I/I_0, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("I/I_0")
    # plt.show()

    # # PLOT I vs x (a single slice)
    # plt.plot(x, I[-1,:])
    # plt.xlabel("x")
    # plt.ylabel("I(x)")
    # plt.title("Intensity profile")
    # plt.show()
