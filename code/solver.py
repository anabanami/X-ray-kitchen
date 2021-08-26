import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import m, cm, mm, nm

plt.rcParams['figure.dpi'] = 150

# folder = Path('Simulation')
# os.makedirs(folder, exist_ok=True)
# os.system(f'rm {folder}/*.png')

# functions

def TIE(z, I, Φ):
    # The intensity and phase evolution of a paraxial monochromatic
    # scalar electromagnetic wave on propagation
    dI_dz = (-1 / k0) * (
        np.real(ifft(1j * k * fft(I)) * ifft(1j * k * fft(Φ)) + I * ifft((1j * k) ** 2 * fft(Φ))
    ))
    return dI_dz

def δ(x, z):
    # refractive index: constant inside the cylinder but zero everywhere else
    δ0 = 462.8 * nm
    # δ_array = np.zeros_like(x * z)
    # δ_array[(z - z_c) ** 2 + (x - x_c) ** 2 <= R ** 2] = δ0

    # refractive index: δ0 within the cylinder decreasing to zero at the edges
    # CDF inspired:
    r = np.sqrt(x**2 + z**2)
    𝜎 = 0.05 * mm
    δ_array = δ0 * (1 / (1 + np.exp((r - R) / 𝜎)))
    return δ_array


def dΨ_dz(z, Ψ):
    # state vector of derivatives in z
    I, Φ = Ψ
    dI_dz = TIE(z, I, Φ)  # how much does this grow per z?
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


def array2D_to_rgb(psi_list):
    '''Takes a 2D array of numbers and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow plots.'''

    I_list, Φ_list = psi_list

    Imax = I_list.max()

    hsv = np.zeros(I_list.shape + (3,), dtype='float')
    hsv[:, :, 0] = Φ_list / (2 * np.pi) % 1
    hsv[:, :, 1] = 1
    hsv[:, :, 2] = np.clip(np.abs(m * I_list) / Imax , 0, 1)

    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    return rgb


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
    Ψ = np.array([I, Φ])

   ########################## RK LOOP ###############################

    # psi_list = []
    # while z < z_final:

    #     print(f"{i = }")

    #     # spatial evolution step
    #     Ψ = Runge_Kutta(z, delta_z, Ψ)
    #     # print(f"\n{Ψ = }")
    #     if not i % 10:
    #         psi_list.append(Ψ)
    #     i += 1
    #     z += delta_z

    # psi_list = np.array(psi_list)
    # print(f"{np.shape(psi_list)}")

    # np.save(f'psi_list.npy', psi_list)

    ######################### PLOTS & TESTS ###############################
    # Load and transpose
    psi_list = np.load("psi_list.npy")
    # psi_list = psi_list.transpose(1, 2, 0) # this returns np.shape(psi_list) = (2, 2048, 2001)

    
    # plt.imshow(array2D_to_rgb(psi_list),
    #             cmap="gist_rainbow",
    #             origin='lower')
    # plt.colorbar()
    # plt.xlabel("z")
    # plt.ylabel("rgb")
    # plt.show()

    # # ###############################

    # I_list, Φ_list = psi_list

    # plt.imshow(I_list,
    #             cmap="gist_rainbow",
    #             origin='lower')
    # plt.colorbar()
    # plt.xlabel("z")
    # plt.ylabel("I")
    # plt.show()

    # ###############################

    # plt.imshow(Φ_list,
    #             cmap="gist_rainbow",
    #             origin='lower')
    # plt.colorbar()
    # plt.xlabel("z")
    # plt.ylabel("Φ")
    # plt.show()

    ########################### OTHER ####################################
    # # z = np.linspace(-x_max, x_max, 2000, endpoint=False).reshape((2000, 1))
    # z = z_c
    # dI_dz, dΦ_dz = dΨ_dz(z, Ψ)

    # # dI_dz Test plot
    # plt.plot(x, dI_dz, label="dI_dz")
    # plt.xlabel("x")
    # plt.ylabel("dI_dz")
    # plt.legend()
    # plt.title(f"dI_dz(x) for {z =:.4f}")
    # plt.show()

    # # # dΦ_dz Test plot
    # plt.plot(x, dΦ_dz, label="dΦ_dz")
    # plt.xlabel("x")
    # plt.ylabel("dΦ_dz")
    # plt.legend()
    # plt.title(f"dΦ_dz(x) for {z =:.4f}")
    # plt.show()
    # ##########

    # # Testing RK near the centre
    # psi_list = np.load("psi_list.npy")
    I, Φ = psi_list[2000]
    # print(np.iscomplex(I)) # array is complex valued
    # print(np.iscomplex(Φ)) # array is real valued

    # # I Test plot
    plt.plot(x, np.real(I), label="I")
    # plt.plot(x, np.imag(I), label="imag I")
    plt.xlabel("x")
    plt.ylabel("I")
    plt.legend()
    plt.title(f"I(x) for {z =:.4f}")
    plt.show()

    # Φ Test plot
    plt.plot(x, np.real(Φ), label="Φ")
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
    # plt.show()
    # ###########