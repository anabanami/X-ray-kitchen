import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import m, cm, mm, nm

plt.rcParams['figure.dpi'] = 150

# functions

def δ(x, z):
    '''Refractive index: δ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    δ0 = 462.8 * nm
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    𝜎 = 0.1 * mm
    δ_array = δ0 * (1 / (1 + np.exp((r - R) / 𝜎)))
    return δ_array # np.shape(δ_array) = (n_x,)


def μ(x, z):
    '''attenuation coefficient: μ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    μ0 = 41.2 # per meter
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    𝜎 = 0.1 * mm
    μ_array = μ0 * (1 / (1 + np.exp((r - R) / 𝜎)))
    return μ_array # np.shape(μ_array) = (n_x,)


def phase(x):
    # phase gain as a function of the cylinder refractive index
    z = np.linspace(-2 * R, 2 * R, 2000, endpoint=False).reshape((2000, 1))
    dz = z[1] - z[0]
    Φ = np.sum(-k0 * δ(x, z) * dz, axis=0)
    return Φ # np.shape(Φ) = (n_x,)


def BLL(x):
    # Brute force integral to find the IC of the intensity (z = z_0)
    z = np.linspace(-2 * R, 2 * R, 2000, endpoint=False).reshape((2000, 1))
    dz = z[1] - z[0]
    I = np.exp(- np.sum(μ(x, z) * dz, axis=0)) * I_0
    return I # np.shape(I) = (n_x,)


def TIE(z, I, Φ):
    '''The intensity and phase evolution of a paraxial monochromatic
    scalar electromagnetic wave on propagation'''
    dI_dz = (-1 / k0) * (
        np.real(
            ifft(1j * k * fft(I)) * ifft(1j * k * fft(Φ))
            + I * ifft((1j * k) ** 2 * fft(Φ))
        )
    )
    return dI_dz # np.shape(dI_dz) = (n_x,)


def Runge_Kutta(z, delta_z, I, Φ):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    k1 = TIE(z, I, Φ)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2, Φ)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2, Φ)
    k4 = TIE(z + delta_z, I + k3 * delta_z, Φ)
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # shape = (n_x,)

def globals():
    # x-array parameters
    x_max = 100 * mm
    x = np.linspace(-x_max, x_max, 2048, endpoint=False)
    delta_x = x[1] - x[0]
    n = x.size

    # X-ray beam parameters
    λ = 0.05166 * nm  # x-rays wavelength
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # Cylinder parameters
    D = 12.75 * mm
    R = D / 2
    z_c = 0 * mm
    x_c = 0 * mm

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, delta_x)


    return x, k0, R, z_c, x_c, k


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, k0, R, z_c, x_c, k = globals()

    # RK Propagation loop parameters
    i = 0
    z = 0
    z_final = 100 * mm
    delta_z = 0.01 * mm  # (n_z = 10000)

    #IC
    I_0 = np.ones_like(x)
    Φ = phase(x)

    ########################## RK LOOP ###############################

    # I_list = []
    # while z < z_final:

    #     print(f"{i = }")

    #     # spatial evolution step
    #     I = Runge_Kutta(z, delta_z, BLL(x), Φ)
    #     if not i % 10:
    #         I_list.append(I)
    #     i += 1
    #     z += delta_z

    # I_list = np.array(I_list)
    # print(f"{np.shape(I_list) = }") #  np.shape(I_list) = (n_z / 10, n_x)


    # np.save(f'I_list.npy', I_list)


    ####################### PLOTS & TESTS #############################
    # Load file
    I_list = np.load("I_list.npy") # np.shape(I_list) = (n_z / 10, n_x)

    I = BLL(x)
    dI_dz = TIE(z, I, Φ) 

    # dI_dz Test plot
    plt.plot(x, dI_dz) # this looks like what I would expect the attenuation factor to look like
    plt.xlabel("x")
    plt.ylabel("dI_dz")
    plt.title(r"TIE: $\frac{\partial I(x)}{\partial z}$ ")
    plt.show()

    # I Test plot
    plt.plot(x, I) # this looks like what I would expect the phase shift to look like
    plt.xlabel("x")
    plt.ylabel("I")
    plt.title(r"Beer-Lamber law: $I(x)$ ")
    plt.show()

    # PLOT ATTENUATION FACTOR I/I0 vs x after RK
    plt.plot(x, I_list[1000] / I_0) # this looks like what I would expect the phase shift to look like
    plt.xlabel("x")
    plt.ylabel(r"$I(x)/I_{0}$")
    plt.title(r"Attenuation factor: $I(x)/I_{0}$ ")
    plt.show()

    # PLOT Φ vs x
    plt.plot(x, Φ)
    plt.xlabel("x")
    plt.ylabel(r"$\phi(x)$")
    plt.title(r"Phase shift $\phi(x) = -k_{0} \int^{z_{0}}_{0} \delta(x, z) dz$ ")
    plt.show()

    # # TODO:
    # # PLOTS ATTENUATION FACTOR I/I0 vs x, z (2D)
