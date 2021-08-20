import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import m, cm, mm, nm

plt.rcParams['figure.dpi'] = 150


folder = Path('dI_dz')
os.makedirs(folder, exist_ok=True)
os.system(f'rm {folder}/*.png')

# functions

def TIE(z, I, Φ):
    # The intensity and phase evolution of a paraxial monochromatic
    # scalar electromagnetic wave on propagation
    dI_dz = (-1 / k0) * (
        ifft(1j * k * fft(I)) * ifft(1j * k * fft(Φ)) + I * ifft((1j * k) ** 2 * fft(Φ))
    )
    return dI_dz


def gaussian2D(x, z, amplitude=1 * mm, centre=0, sigma=1 * mm):
    return (
        amplitude
        * (1 / (sigma * (np.sqrt(2 * np.pi))))
        * (np.exp((-1 / 2) * ((((x - centre) ** 2 + (z - centre) ** 2) / sigma) ** 2)))
    )


def δ(x, z):
    # refractive index: constant inside the cylinder but zero everywhere else
    δ0 = 462.8 * nm
    δ_array = np.zeros_like(x * z)
    # δ_array[(z - z_c) ** 2 + (x - x_c) ** 2 <= R ** 2] = δ0
    δ_array[gaussian2D(x, z) >= R ** 2] = δ0

    # print(f"\n{np.shape(δ_array) = }\n")
    return δ_array


def dΨ_dz(z, Ψ):
    # state vector of derivatives in z
    I, Φ = Ψ
    dI_dz = TIE(z, I, Φ)  # how much does this grow per z?

    # if not i % 1000:
    #     # # dI_dz Test plot
    #     plt.plot(x, dI_dz, label="dI_dz")
    #     plt.xlabel("x")
    #     plt.ylabel("dI_dz")
    #     plt.legend()
    #     plt.title(f"dI_dz(x) for {z =:.4f}")
    #     plt.savefig(folder/f"{i:04d}")
    #     plt.show()
    #     plt.clf()

    dΦ_dz = -k0 * δ(x, z)  # how much does this grow per z?

    return np.array([dI_dz, dΦ_dz])


def Runge_Kutta(z, delta_z, Ψ):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    # Ψ is array with shape: (2, 2048)
    k1 = dΨ_dz(z, Ψ)
    k2 = dΨ_dz(z + delta_z / 2, Ψ + k1 * delta_z / 2)
    k3 = dΨ_dz(z + delta_z / 2, Ψ + k2 * delta_z / 2)
    k4 = dΨ_dz(z + delta_z, Ψ + k3 * delta_z)
    return Ψ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # array shape = (2, 2048)


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
    R = 12.75 / 2 * mm
    z_c = 0 * mm
    x_c = 0 * mm

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, delta_x)

    # Propagation & loop parameters
    i = 0
    z_max = 100 * mm
    z = -z_max
    z_final = z_max
    delta_z = 0.01 * mm  # (n_z = 20000)

    ######################### TESTING ###############################

    # ICs
    I = np.ones_like(x)
    Φ = np.zeros_like(x)
    # Φ = gaussian(x) # testing nice smooth IC

    Ψ = np.array([I, Φ])

    ######################### RK LOOP ###############################

    # psi_list = []
    # while z < z_final:

    #     print(f"{i = }")

    #     # spatial evolution step
    #     Ψ = Runge_Kutta(z, delta_z, Ψ)
    #     # print(f"\n{Ψ = }")

    #     psi_list.append(Ψ)
    #     i += 1
    #     z += delta_z

    # psi_list = np.array(psi_list)
    # np.save(f'TIE/psi_list.npy', psi_list)

    ######################### PLOTS & TESTS ###############################

    # # z = np.linspace(-x_max, x_max, 2000, endpoint=False).reshape((2000, 1))
    # z = z_c
    # dI_dz, dΦ_dz = dΨ_dz(z, Ψ)

    # # dI_dz Test plot
    # plt.plot(x, dI_dz, label="dI_dz")
    # # plt.xlim(-20 * mm, 20 * mm)
    # plt.xlabel("x")
    # plt.ylabel("dI_dz")
    # plt.legend()
    # plt.title(f"dI_dz(x) for {z =:.4f}")
    # # plt.savefig(folder/f"{i:04d}")
    # plt.show()
    # # plt.clf()

    # # # dΦ_dz Test plot
    # plt.plot(x, dΦ_dz, label="dΦ_dz")
    # plt.xlim(-20 * mm, 20 * mm)
    # plt.xlabel("x")
    # plt.ylabel("dΦ_dz")
    # plt.legend()
    # plt.title(f"dΦ_dz(x) for {z =:.4f}")
    # # plt.savefig(folder2/f"{i:04d}")
    # plt.show()
    # # plt.clf()
    # ##########

    # # Testing RK near the centre
    psi_list = np.load("TIE/psi_list.npy")
    I, Φ = psi_list[9363]

    # # I Test plot
    plt.plot(x, I, label="I")
    # plt.xlim(-20 * mm, 20 * mm)
    plt.xlabel("x")
    plt.ylabel("I")
    plt.legend()
    plt.title(f"I(x) for {z =:.4f}")
    plt.show()

    # Φ Test plot
    plt.plot(x, Φ, label="Φ")
    # plt.xlim(-10 * mm, 10 * mm)
    plt.xlabel("x")
    plt.ylabel("Φ")
    plt.legend()
    plt.title(f"Φ(x) for {z =:.4f}")
    plt.show()
    ##########

    # # ###########
    # ## PLAYING AROUND with δ ###
    # z = np.linspace(-x_max, x_max, 2000, endpoint=False).reshape((2000, 1))
    # δ_array = δ(x, z)
    # plt.imshow(δ_array, origin='lower')
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel(r"$\delta(x, z)$")
    # # plt.legend()
    # plt.show()
    # ###########

    ########################### OTHER ####################################

    # # gaussian Test plot
    # # print(f"\n{np.shape(gaussian2D(x, z)[1, :]) = }\n")
    # plt.plot(x, gaussian2D(x, z)[1, :], label="gaussian2D")
    # plt.xlabel("x")
    # plt.legend()
    # plt.show()
    # # plt.clf()
    ##########