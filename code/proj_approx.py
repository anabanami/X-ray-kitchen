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
    decreasing to zero at the edges CDF inspired:'''
    δ0 = 462.8 * nm
    r = np.sqrt(x ** 2 + z ** 2)
    𝜎 = 0.05 * mm
    δ_array = δ0 * (1 / (1 + np.exp((r - R) / 𝜎)))
    return δ_array # np.shape(δ_array) = (2048,)


def μ(x, z):
    '''attenuation coefficient: μ0 within the cylinder 
    decreasing to zero at the edges CDF inspired:'''
    μ0 = 41.2 # per meter
    r = np.sqrt(x ** 2 + z ** 2)
    𝜎 = 0.05 * mm
    μ_array = μ0 * (1 / (1 + np.exp((r - R) / 𝜎)))
    return μ_array # np.shape(μ_array) = (2048,)


def phase(x, R):
    # phase gain as a function of the cylinder refractive index
    z = np.linspace(z_initial, z_final, 2000, endpoint=False).reshape((2000, 1))
    dz = z[1] - z[0]
    Φ = np.sum(-k0 * δ(x, z) * dz, axis=0)
    return Φ # np.shape(Φ) = (n_x,) = (2048,)


def BLL(x, z):
    # Brute force integral to find the IC of the intensity (z = z_0)
    z_array = np.linspace(z_initial, z_final, 2000, endpoint=False).reshape((2000, 1))
    dz = z_array[1] - z_array[0]
    I = np.exp(- np.sum(μ(x, z) * delta_z, axis=0)) * I_0
    return I # np.shape(I) = (n_x,) = ()


def TIE(z, I, Φ):
    '''The intensity and phase evolution of a paraxial monochromatic
    scalar electromagnetic wave on propagation'''
    dI_dz = (-1 / k0) * (
        np.real(
            ifft(1j * k * fft(I)) * ifft(1j * k * fft(Φ))
            + I * ifft((1j * k) ** 2 * fft(Φ))
        )
    )
    return dI_dz # what shape is dis?


def Runge_Kutta(z, delta_z, I):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    # I is array with shape: (2048,)
    k1 = TIE(z, I, Φ)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2, Φ)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2, Φ)
    k4 = TIE(z + delta_z, I + k3 * delta_z, Φ)
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # what shape is dis?


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    # x-array parameters
    x_max = 100 * mm
    x = np.linspace(-x_max, x_max, 2048, endpoint=False)
    n = x.size
    delta_x = x[1] - x[0]

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

    # Propagation & loop parameters
    i = 0
    z_initial = 0
    z_final = 100 * mm
    delta_z = 0.01 * mm  # (n_z = 20000)

    #IC
    I_0 = np.ones_like(x) # 

    
    ########################## RK LOOP ###############################

    # Φ = phase(x, R)

    # I_list = []
    # while z_initial < z_final:

    #     print(f"{i = }")

    #     # spatial evolution step
    #     I = Runge_Kutta(z_initial, delta_z, BLL(x, z_initial))
    #     # print(f"\n{I = }")
    #     # print(f"{np.shape(I) = }")

    #     if not i % 10:
    #         I_list.append(I)
    #     i += 1
    #     z_initial += delta_z

    # I_list = np.array(I_list)
    # print(f"{np.shape(I_list) = }") #  np.shape(I_list) = (1001, 2048)


    # np.save(f'I_list.npy', I_list)

    ####################### PLOTS & TESTS #############################

    # print(f"\n{np.shape(δ(x, z)) = }")
    # print(f"\n{np.shape(μ(x, z)) = }")
    # print(f"\n{np.shape(phase(x, z, R)) = }")
    # print(f"\n{np.shape(BLL(x, z)) = }")

    # Load and transpose?
    I_list = np.load("I_list.npy") # np.shape(I_list) = (1001, 2048)
    # # I_list = I_list.transpose(1, 0) # np.shape(I_list) = (2048, 1001) ?????
    # # print(f"{np.shape(I_list) = }")  


    # # TODO:
    # # PLOTS ATTENUATION FACTOR I/I0 vs x (in 1D) and vs x, z (2D)
    plt.plot(x, I_list[1000] / I_0, label="I")
    plt.xlabel("x")
    plt.ylabel("I")
    plt.legend()
    plt.title(f"I(x)")
    plt.show()




    # PLOT Φ vs x

