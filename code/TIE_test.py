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

folder2 = Path('dΦ_dz')
os.makedirs(folder2, exist_ok=True)
os.system(f'rm {folder2}/*.png')

folder3 = Path('phase-shift')
os.makedirs(folder3, exist_ok=True)
os.system(f'rm {folder3}/*.png')

folder4 = Path('intensity')
os.makedirs(folder4, exist_ok=True)
os.system(f'rm {folder4}/*.png')

# functions

def TIE(z, I, Φ):
    # The intensity and phase evolution of a paraxial monochromatic
    # scalar electromagnetic wave on propagation

    dI_dz = (-1 / k0) * (
        ifft(1j * k * fft(I)) * ifft(1j * k * fft(Φ)) + I * ifft((1j * k) ** 2 * fft(Φ))
    )
    # print(f"\n{all(dI_dz == 0) = }")
    return dI_dz

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
    dI_dz = TIE(z, I, Φ) # how much should this grow per z?
    dΦ_dz = -k0 * δ(x, z) # how much should this grow per z?
    return np.array([dI_dz, dΦ_dz])

def Runge_Kutta(z, delta_z, Ψ):
    # spatial evolution 4th order RK
    # z is single value
    # Ψ is array with shape: (2, 2048)
    # print(f"\n{np.shape(Ψ) = }\n")
    k1 = dΨ_dz(z, Ψ)
    k2 = dΨ_dz(z + delta_z / 2, Ψ + k1 * delta_z / 2)
    k3 = dΨ_dz(z + delta_z / 2, Ψ + k2 * delta_z / 2)
    k4 = dΨ_dz(z + delta_z, Ψ + k3 * delta_z)
    return Ψ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # array shape = (2, 2048)

def complex_array_to_rgb(Ψ, i):  # , rmax=None):
    '''Takes an array of complex numbers and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''

    I, Φ = Ψ
    absmax = np.abs(I).max()
    # print(f"\n{absmax = } ")
    hsv = np.zeros(Ψ.shape + (3,), dtype='float')
    # print(f"\n{hsv = }")
    hsv[:, :, 0] = Φ / (2 * np.pi) % 1
    hsv[:, :, 1] = 1
    hsv[:, :, 2] = np.clip(np.abs(I) / absmax, 0, 1)  # I/I.max() #
    # print(f"\n{hsv =  }")
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    # print(f"\n{rgb = }")
    return rgb


# -------------------------------------------------------------------------------- #

if __name__ == '__main__':

    # x-array parameters
    x_max = 100 * mm
    x_min = - x_max
    x = np.linspace(x_min, x_max, 2048, endpoint=False)
    n = x.size
    x_step = x[1] - x[0]

    # X-ray beam parameters
    λ = 0.05166 * nm  # x-rays wavelength
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # Cylinder parameters
    R = 12.75 / 2 * mm
    z_c = 0 * mm
    x_c = 0 * mm

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # Propagation & loop parameters
    i = 0
    z_max = 100 * mm
    z = -z_max
    z_final = z_max
    delta_z = 0.01 * mm # 0.01 * mm # 1 * mm ###### not many steps atm (n_z = 1000)

    ######################### evolution algorithm ###############################

    # ICs
    I = np.ones_like(x)
    Φ = np.zeros_like(x)
    
    Ψ = np.array([I, Φ])

    psi_list = []
    while z < z_final:

        print(f"{i = }")

        # spatial evolution step
        Ψ = Runge_Kutta(z, delta_z, Ψ)
        # print(f"\n{Ψ = }")

        psi_list.append(Ψ)
        i += 1
        z += delta_z

    psi_list = np.array(psi_list)
    # print(f"\n{np.shape(psi_list) = }") #(n_z, 2, n_x) = (n_z, 2, 2048)

    ################################### TESTS ###################################
    ## PLAYING AROUND with psi_list (I) ###
    ## print(f"\n{np.shape(psi_list) = }")
    
    # for j, psi in enumerate(psi_list):
    #     # print(f"\n{np.shape(psi) = }")
    #     # print(f"\n{psi = }")
    #     # I unpack the state vector to visualise the intensity
    #     I, Φ = psi
    #     # print(f"\n{np.shape(I) = }")
    #     if not j % 1000:
    #         plt.plot(x, I)
    #         plt.xlabel("x")
    #         plt.ylabel("I(x, z)")
    #         plt.savefig(folder/f'{j:04d}.png')
    #         # plt.show()
    #         plt.clf()


    ### PLAYING AROUND with dΨ_dz(z, Ψ) ###
    # dΨ_dz(-6.4 * mm, Ψ)
    # dΨ_dz(-6.35 * mm, Ψ)
    # dΨ_dz(-5.5 * mm, Ψ)
    # dΨ_dz(-1.75 * mm, Ψ)
    # dΨ_dz(-0.5 * mm, Ψ)
    # dΨ_dz(-0.25 * mm, Ψ)
    # dΨ_dz(0, Ψ)
    # dΨ_dz(0.25 * mm, Ψ)
    # dΨ_dz(0.5 * mm, Ψ)
    # dΨ_dz(1.75 * mm, Ψ)
    # dΨ_dz(5.5 * mm, Ψ)
    # dΨ_dz(6 * mm, Ψ)
    # dΨ_dz(6.35 * mm, Ψ)
    # dΨ_dz(6.4 * mm, Ψ)

    ############
    # z = np.linspace(-z_max, z_max, 2000, endpoint=False).reshape((2000, 1))
    # z = z_c
    # ## PLAYING AROUND with Φ ###
    # # I unpack the state vector to visualise the phase
    # I, Φ = Ψ
    # # phase test PLOT #
    # # if not i % 100:
    # plt.plot(x, Φ[:], label="Φ")
    # plt.xlim(-20 * mm, 20 * mm)
    # plt.xlabel("x")
    # plt.ylabel("Φ")
    # plt.legend()
    # plt.title(f"Φ(x) for z = {z:.4f}")
    # plt.savefig(folder3/f'{i:04d}.png')
    # # plt.show()
    # plt.clf()

    # ## PLAYING AROUND with dI_dz ###
    # print(f"\n{np.shape(dI_dz) = }") # this returns: (2048,)
    # # I unpack the state vector to visualise the dI_dz
    # I, Φ = Ψ
    # # dI_dz Test plot
    # if not i % 10:
    #     plt.plot(x, dI_dz, label="dI_dz")
    #     plt.xlim(-20 * mm, 20 * mm)
    #     plt.xlabel("x")
    #     plt.ylabel("dI_dz")
    #     plt.legend()
    #     plt.title(f"dI_dz(x) for {z =:.4f}")
    #     plt.savefig(folder4/f"{i:04d}")
    #     # plt.show()
    #     plt.clf()
    # ###########


    ################################### TODO ###################################
    # # Simulation frames
    # plt.imshow(
    #     complex_array_to_rgb(Ψ, i),
    #     cmap="gist_rainbow",
    #     origin='lower',
    #     extent=(-0.2 * x_max, 0.2 * x_max, -0.2 * x_max, 0.2 * x_max),
    # )
    # plt.savefig(folder/f'{i:04d}.png')
    # # plt.show()
    # plt.clf()


