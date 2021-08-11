import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import m, mm, nm

plt.rcParams['figure.dpi'] = 200

folder = Path('TIE')
# os.makedirs(folder, exist_ok=True)
# os.system(f'rm {folder}/*.png')

# functions
def TIE(z, I, Φ):
    # propagating in space
    return -(1 / k0) * (2 * ifft(2 * np.pi * (-k ** 2) * fft(I) * fft(Φ)))

def δ(x, z):
    # refractive index: constant inside the cylinder but zero everywhere else
    δ0 = 462.8 * nm
    δ_array = np.zeros_like(x * z)
    δ_array[(z - z_c) ** 2 + (x - x_c) ** 2 <= R ** 2] = δ0
    # print(f"\n{np.shape(δ_array) = }\n")
    return δ_array

def dΨ_dz(z, Ψ):
    # state vector of derivatives in z
    I, Φ = Ψ
    dI_dz = TIE(z, I, Φ)
    dΦ_dz = -k * δ(x, z)
    return np.array([dI_dz, dΦ_dz])

def Runge_Kutta(z, delta_z, Ψ):
    # spatial evolution 4th order RK
    k1 = dΨ_dz(z, Ψ)
    k2 = dΨ_dz(z + delta_z / 2, Ψ + k1 * delta_z / 2)
    k3 = dΨ_dz(z + delta_z / 2, Ψ + k2 * delta_z / 2)
    k4 = dΨ_dz(z + delta_z, Ψ + k3 * delta_z)
    return Ψ + (delta_z / 6) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )  # what is this doing????? Ψ + ___ ???


# -------------------------------------------------------------------------------- #

if __name__ == '__main__':

    # x- array parameters
    x_max = 1
    x = np.linspace(0, x_max, 2048, endpoint=False)
    n = x.size
    x_step = x[1] - x[0]

    # X-ray beam parameters
    λ = 0.05166 * nm  # x-rays wavelength
    k0 = 2 * np.pi / λ

    # For Fourier space (x dimension only)
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # Propagation parameters & loop
    i = 0
    z = 0 * m
    z_final = 1 * m
    delta_z = 0.001 * m ###############################

    # Cylinder parameters
    R = 12.75 / 2 * mm
    z_c = 0.5 * m
    x_c = 0.5 * m

    # ICs
    # I = np.ones_like(x)
    # Φ = np.zeros_like(x)
    # ICs (Projection approximation)
    I = np.ones_like(x)
    Φ = np.zeros_like(x)

    Ψ = np.array([I, Φ])

    # while z < z_final:
    #     print(i)
    #     # spatial evolution step
    #     Ψ = Runge_Kutta(z, delta_z, Ψ)
    #     i += 1
    #     z += delta_z

    # # After the integration occurs I unpack the state vector
    # I, Φ = Ψ

    # TEST PLOTS
    ### PLAYING AROUND with δ ###
    z = np.linspace(0, x_max, 2000, endpoint=False).reshape((2000, 1))
    δ_array = δ(x, z)
    plt.imshow(δ_array, origin='lower')
    plt.xlabel("x")
    plt.ylabel("δ")
    plt.show()
    ### ------------ ###

    # # SHOULD I VISUALISE THESE?
    # # constant (I = 1 vs x)??
    # plt.plot(x, I)
    # plt.xlabel("x")
    # plt.ylabel("I")
    # plt.show()

    # # zig zag ??
    # plt.plot(x, np.real(Φ), label="real Φ")
    # # plt.plot(x, np.imag(Φ), label="imaginary Φ")
    # plt.xlabel("x")
    # plt.ylabel("Φ")
    # plt.legend()
    # plt.show()

# # -------------------------------------------------------------------------------- #

# ## TEST PLOTS
# # if not i % 100:
# #     plt.plot(x, np.real(Φ), label="real Φ")
# #     plt.plot(x, np.imag(Φ), label="imaginary Φ")
# #     plt.xlim(-x_max, x_max)
# #     plt.legend()
# #     plt.xlabel("x")
# #     plt.ylabel("Φ")
# #     # plt.title(f"")
# #     plt.savefig(folder/f'{i:04d}.png')
# #     plt.show()
# #     plt.clf()
