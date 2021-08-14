import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import m, cm, mm, nm

plt.rcParams['figure.dpi'] = 100

folder = Path('TIE')
os.makedirs(folder, exist_ok=True)
os.system(f'rm {folder}/*.png')

# functions
def TIE(z, I, Φ):
    # propagating in space
    return (- 1 / k0) * (2 * ifft(4 * np.pi**2 * (-k ** 2) * fft(I) * fft(Φ)))

def δ(x, z):
    # refractive index: constant inside the cylinder but zero everywhere else
    δ0 = 462.8 * nm
    δ_array = np.zeros_like(x * z)
    δ_array[(z - z_c) ** 2 + (x - x_c) ** 2 <= R ** 2] = δ0
    # print(f"\n{np.shape(δ_array) = }\n")
    return δ_array

# def dΨ_dz(z, Ψ):
#     # state vector of derivatives in z
#     I, Φ = Ψ
#     dI_dz = TIE(z, I, Φ)
#     dΦ_dz = -k0 * δ(x, z)
#     # plot
#     return np.array([dI_dz, dΦ_dz])

def dΨ_dz(z, Ψ):
    # state vector of derivatives in z
    I, Φ = Ψ
    dI_dz = TIE(z, I, Φ)
    dΦ_dz = -k0 * δ(x, z)
    # print(f"\n{np.shape(dΦ_dz[:]) = }\n") # this returns: np.shape(δ_array) = (2048,)
    # test PLOT #
    # how much should this grow per z?
    plt.plot(x, dΦ_dz[:], label="dΦ_dz") 
    plt.xlabel("x")
    plt.ylabel("dΦ_dz")
    plt.legend()
    plt.title(f"dΦ_dz(x) for z = {z}")
    plt.savefig(folder/f'z ={z}.png')
    # plt.show()
    plt.clf()
    return np.array([dI_dz, dΦ_dz])

def Runge_Kutta(z, delta_z, Ψ):
    # spatial evolution 4th order RK
    # z is single value
    # Ψ is array with shape: (2, 2048)
    print(f"\n{np.shape(Ψ) = }\n")

    k1 = dΨ_dz(z, Ψ) # array
    print(f"\n{np.shape(k1) = }\n")

    print(f"\n{np.shape(Ψ + k1 * delta_z) = } \n")

    k2 = dΨ_dz(z + delta_z / 2, Ψ + k1 * delta_z / 2) # array
    # print(f"\n{np.shape(k2) = } \n")

    k3 = dΨ_dz(z + delta_z / 2, Ψ + k2 * delta_z / 2) # array
    # print(f"\n{np.shape(k3) = }\n")

    k4 = dΨ_dz(z + delta_z, Ψ + k3 * delta_z) # array
    # print(f"\n{np.shape(k4) = }\n")

    # print(f"\n{np.shape(Ψ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)) = }\n")
    return Ψ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4) # array shape (2, 2048)


# -------------------------------------------------------------------------------- #

if __name__ == '__main__':

    # x-array parameters
    x_max = 100 * mm
    x_min = - 100 * mm
    x = np.linspace(x_min, x_max, 2048, endpoint=False)
    n = x.size
    x_step = x[1] - x[0]

    # X-ray beam parameters
    λ = 0.05166 * nm  # x-rays wavelength
    k0 = 2 * np.pi / λ

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # Propagation & loop parameters
    i = 0
    z = 0 * mm
    z_final = x_max
    delta_z = 1 * mm

    # Cylinder parameters
    R = 12.75 / 2 * mm
    z_c = 0 * mm
    x_c = 0 * mm

    ################### evolution algorithm #############################

    # ICs
    I = np.ones_like(x)
    Φ = np.zeros_like(x)
    # Initial state vector
    Ψ = np.array([I, Φ])

    # while z < z_final:

    #     print(f"{i = }")

    #     # # TEST PLOT
    #     # zig zag ??
    #     if not i % 10:
    #         plt.plot(x, np.real(Φ), label="real Φ")
    #         plt.plot(x, np.imag(Φ), label="imaginary Φ")
    #         plt.xlabel("x")
    #         plt.ylabel("Φ")
    #         plt.legend()
    #         plt.title(f"Φ(x) for z = {z:.03f}")
    #         plt.savefig(folder/f'z ={i:04d}.png')
    #         # plt.show()
    #         plt.clf()

    #     # spatial evolution step
    #     Ψ = Runge_Kutta(z, delta_z, Ψ)
    #     i += 1
    #     z += delta_z

    # # After the integration occurs I unpack the state vector
    # I, Φ = Ψ

    ####################### useful ###################################

    # # TEST PLOTS
    # ### PLAYING AROUND with δ ###
    # z = np.linspace(0, x_max, 2000, endpoint=False).reshape((2000, 1))
    # δ_array = δ(x, z)
    # plt.imshow(δ_array, origin='lower')
    # plt.xlabel("x")
    # plt.ylabel("δ")
    # plt.show()


    ### PLAYING AROUND with dΨ_dz(z, Ψ) ###
    dΨ_dz(-1*mm, Ψ)
    dΨ_dz(-0.5*mm, Ψ)
    dΨ_dz(0, Ψ)
    dΨ_dz(0.5*mm, Ψ)
    dΨ_dz(1*mm, Ψ)
    # ### ------------ ###



    # constant (I = 1 vs x)??
    # plt.plot(x, I)
    # plt.xlabel("x")
    # plt.ylabel("I")
    # plt.show()