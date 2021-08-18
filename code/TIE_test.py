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
    # print(f"{(- 1 / k0) * (2 * ifft(4 * np.pi**2 * (-k ** 2) * fft(I) * fft(Φ)))}")
    return np.real((- 1 / k0) * (2 * ifft(4 * np.pi**2 * (-k ** 2) * fft(I) * fft(Φ))))

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
    dΦ_dz = -k0 * δ(x, z)
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
    return Ψ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4) # array shape = (2, 2048)

def complex_array_to_rgb(Ψ, i): #, rmax=None):
    '''Takes an array of complex numbers and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''

    I, Φ = Ψ

    absmax = np.abs(Ψ).max()
    print(f"\n{absmax = } ")

    hsv = np.zeros(Ψ.shape + (3,), dtype='float')
    print(f"\n{hsv = }")
    hsv[:, :, 0] = Φ / (2 * np.pi) % 1
    print(f"\n{hsv[:, :, 0] = }")

    hsv[:, :, 1] = 1
    print(f"\n{hsv[:, :, 1] = }")

    hsv[:, :, 2] = np.clip(np.abs(Ψ) / absmax, 0, 1) # I/I.max() #
    print(f"\n{hsv[:, :, 2] = }")

    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

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
    k0 = 2 * np.pi / λ # x-rays wavenumber

    # For Fourier space
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # Propagation & loop parameters
    i = 0
    z = -x_max
    z_final = x_max
    delta_z = 1 * mm # 0.01 * mm

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

    plt.imshow(complex_array_to_rgb(Ψ, i), cmap="gist_rainbow", origin='lower')
    # plt.savefig(folder/f'{i:04d}.png')
    plt.show()
    plt.clf()
    while z < z_final:

        print(f"{i = }")
        # # Simulation frames
        # plt.imshow(complex_array_to_rgb(Ψ, i), cmap="gist_rainbow", origin='lower')
        # plt.savefig(folder/f'{i:04d}.png')
        # # plt.show()
        # plt.clf()

        # spatial evolution step
        Ψ = Runge_Kutta(z, delta_z, Ψ)
        i += 1
        z += delta_z

    # After the integration occurs I unpack the state vector
    I, Φ = Ψ
    # Simulation frames
    plt.imshow(complex_array_to_rgb(Ψ, i), cmap="gist_rainbow", origin='lower')
    # plt.savefig(folder/f'{i:04d}.png')
    plt.show()
    plt.clf()

    ####################### TESTS ###################################

    # TEST PLOTS
    ### PLAYING AROUND with δ ###
    # z = np.linspace(-x_max, x_max, 2000, endpoint=False).reshape((2000, 1))
    # δ_array = δ(x, z)
    # plt.imshow(δ_array, origin='lower')
    # plt.xlabel("x")
    # plt.ylabel("δ")
    # # plt.legend()
    # plt.show()

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

    ### PLAYING AROUND with Φ ###
        # # I unpack the state vector to visualise the phase
        # I, Φ = Ψ
        # # phase test PLOT #
        # if not i % 100:
        #     plt.plot(x, Φ[:], label="Φ") 
        #     plt.xlim(-20 * mm, 20 * mm)
        #     plt.xlabel("x")
        #     plt.ylabel("Φ")
        #     plt.legend()
        #     plt.title(f"Φ(x) for z = {z:.4f}")
        #     plt.savefig(folder/f'{i:04d}.png')
        #     # plt.show()
        #     plt.clf()

    ### testing if TIE() has non zero imaginary part ###
    # TIE(z, I, Φ)



    ### ------------ ###

    # # I vs x??
    # plt.plot(x, I)
    # plt.xlabel("x")
    # plt.ylabel("I")
    # plt.show()