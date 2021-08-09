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
os.makedirs(folder, exist_ok=True)
os.system(f'rm {folder}/*.png')

# functions
def TIE(z, I, Φ):
    # propagating in free space
    return - (1 / k) * (2 * ifft(2 * np.pi * (- k**2) * np.convolve(fft(I), fft(Φ), mode='same')))

def δ(x_array, z_value):
    # refractive index: constant inside the cylinder but zero everywhere else
    δ0 = 462.8 * nm
    δ_array = np.zeros_like(x_array)
    for i in range(len(x_array)):
        δ_array[(z_value - z_c)**2 + (x_array[i] - x_c)**2 <= R**2] = δ0
        # print(f"{δ_array = }\n")

def dΨ_dz(z, Ψ):
    # state vector of derivatives in z
    I, Φ = Ψ
    dI_dz = TIE(z, I, Φ)
    dΦ_dz = - k * δ(x, z)
    return np.array([dI_dz, dΦ_dz])

def Runge_Kutta(z, delta_z, Ψ):
    # spatial evolution 4th order RK
    k1 = dΨ_dz(z, Ψ)
    k2 = dΨ_dz(z + delta_z / 2, Ψ + k1 * delta_z / 2) 
    k3 = dΨ_dz(z + delta_z / 2, Ψ + k2 * delta_z / 2) 
    k4 = dΨ_dz(z + delta_z, Ψ + k3 * delta_z) 
    return Ψ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == '__main__':

    # x- array parameters
    x_max = 5
    x = np.linspace(-x_max, x_max, 1024, endpoint=False)
    n = x.size
    x_step = x[1] - x[0]

    # X-ray beam parameters
    λ = 0.01 * nm # x-rays wavelength
    k0 = 2 * np.pi / λ 

    # For Fourier space (one dimension only)
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # Propagation parameters & loop
    i = 0
    z = 0
    z_final = 1 * m
    delta_z = 0.1 * m # correct this soon! make step smaller

    # circle parameters
    R = 12.75 / 2 * mm
    z_c = 0.5 * m
    x_c = 0.5 * m

    # ICs
    I = np.ones_like(x)
    Φ = np.zeros_like(x)

    state_vector = [I, Φ]

    while z < z_final:

        # TEST PLOTS
        if not i % 500:
            plt.plot(x, np.real(I), label="real I")
            plt.plot(x, np.imag(I), label="imaginary I")
            plt.plot(x, np.real(Φ), label="real Φ")
            plt.plot(x, np.imag(Φ), label="imaginary Φ")
            plt.xlim(-x_max, x_max)
            plt.legend()
            plt.xlabel("z")
            plt.ylabel("x")
            # plt.title(f"")
            plt.savefig(folder/f'{i:04d}.png')
            # plt.show()
            plt.clf()
           
        # Evolution step
        state_vector = Runge_Kutta(z, delta_z, state_vector)

        # δ(x, z)
        
        i += 1
        z += delta_z
