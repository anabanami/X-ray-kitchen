import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import nm

plt.rcParams['figure.dpi'] = 200

folder = Path('TIE')
os.makedirs(folder, exist_ok=True)
os.system(f'rm {folder}/*.png')

# functions
def TIE(z, I, Φ):
    # propagating in free space
    return 2 * ifft(2 * np.pi * (- k**2) * np.convolve(fft(I), fft(Φ), mode='same'))

def Runge_Kutta(z, delta_z, I, Φ):
    k1 = TIE(z, I, Φ)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2, Φ + k1 * delta_z / 2) 
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2 , Φ + k2 * delta_z / 2) 
    k4 = TIE(z + delta_z, I + k3 * delta_z, Φ + k3 * delta_z) 
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4), Φ + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == '__main__':

    x_max = 5
    x = np.linspace(-x_max, x_max, 1024, endpoint=False)
    n = x.size
    x_step = x[1] - x[0]

    λ = 1 * nm # soft x-rays wavelength
    k0 = 2 * np.pi / λ 
    # For Fourier space
    # one dimension only
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # Propagation loop
    i = 0
    z = 0
    z_final = 5
    delta_z = 0.001

    # ICs
    I = np.ones_like(x)
    Φ = np.zeros_like(x)

    while z < z_final:

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
           
        #z derivatives! 
        I, Φ = Runge_Kutta(z, delta_z, I, Φ)
        i += 1
        z += delta_z
