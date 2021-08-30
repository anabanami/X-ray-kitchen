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

def y_sigmoid(y):
    𝜎 = 0.1 * mm
    S = (1 / (1 + np.exp(np.abs(-(y - y_c)) / 𝜎)))
    return S # np.shape = (n_y,)


if __name__ == '__main__':

    # y-array parameters
    n_y = 2000
    y_max = 100 * mm
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)
    delta_y = y[1] - y[0]
    n = y.size

    # Cylinder parameters
    y_c = 0 * mm

    # dI_dz Test plot
    plt.plot(y, y_sigmoid(y))
    plt.xlabel("y")
    plt.ylabel("S(y)")
    plt.title("S(y) ")
    plt.show()


