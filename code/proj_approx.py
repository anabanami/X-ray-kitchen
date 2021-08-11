import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft
from physunits import m, mm, nm

plt.rcParams['figure.dpi'] = 200

# functions
def t(x, z):
    t_array = np.zeros_like(x * z)
    t_array[(z - z_c) ** 2 + (x - x_c) ** 2 <= R ** 2] = 2 # this line doesn't do anything atm!
    print(f"\n{t_array = }\n") 
    return t_array

def TIE(z, I, Φ):
    return -(1 / k0) * (2 * ifft(2 * np.pi * (-k ** 2) * fft(I) * fft(Φ)))


# -------------------------------------------------------------------------------- #

if __name__ == '__main__':

    # x- array parameters
    x_max = 1
    x = np.linspace(0, x_max, 2048, endpoint=False)
    n = x.size
    x_step = x[1] - x[0]

    # z- array parameters
    z = np.linspace(0, x_max, 2000, endpoint=False).reshape((2000, 1))

    # X-ray beam parameters
    λ = 0.05166 * nm  # x-rays wavelength
    k0 = 2 * np.pi / λ
    I0 = np.ones_like(x) #???

    # For Fourier space (x dimension only)
    k = 2 * np.pi * np.fft.fftfreq(n, x_step)

    # cylinder parameters
    R = 12.75 / 2 * mm
    z_c = 0.5 * m
    x_c = 0.5 * m
    μ = 41.2 / m
    δ = 462.8 * nm

    # ICs (Projection approximation)
    Φ = np.exp(-μ * t(x, z))
    I = I0 * Φ
    
    # using projection approximation in TIE
    I = TIE(z, I, Φ)
    print(f"{I = }") # this is an array of zeros ???



