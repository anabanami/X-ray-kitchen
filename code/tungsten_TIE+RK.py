import numpy as np
from scipy.ndimage.filters import laplace
import scipy.constants as const
from physunits import m, cm, mm, nm, um, keV
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

plt.rcParams['figure.dpi'] = 150

def thicc(x, R):
    # # Create cylindrical object projected thickness
    T = np.zeros_like(x)
    T[0:100] = 1
    T = 2 * np.sqrt(R ** 2 - x ** 2)
    T = np.nan_to_num(T)
    T = gaussian_filter(T, sigma=4)
    return T


def gradΦ_laplacianΦ(Φ):
    dΦ_dx = np.gradient(Φ, delta_x, axis=0)
    lap_Φ = laplace(Φ / delta_x**2)
    return dΦ_dx, lap_Φ


def TIE(z, I):
    '''The intensity and phase evolution of a paraxial monochromatic
    scalar electromagnetic wave on propagation (1D)'''
    dI_dx = np.gradient(I, delta_x, axis=0)
    dI_dz = (-1 / k) * (
        dI_dx * dΦ_dx +
        I * lap_Φ
        )
    return dI_dz  # np.shape(dI_dz) = (n_x,)


def Runge_Kutta(z, delta_z, I):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    k1 = TIE(z, I)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2)
    k4 = TIE(z + delta_z, I + k3 * delta_z)    
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # shape = (n_x)


def propagation_loop(I_0):
    # RK Propagation loop parameters
    i = 0
    z = 0

    I = I_0
    I_list = []
    print("<< propagating wavefield >>") 
    while z < z_final:
        print(f"{i = }")
        # spatial evolution step
        I = Runge_Kutta(z, delta_z, I)
        if not i % 10:
            I_list.append(I)
        i += 1
        z += delta_z

    I_list = np.array(I_list)
    print(f"{np.shape(I_list) = }") #  np.shape(I_list) = (n_z / 10, n_x)
    # np.save(f'I_list.npy', I_list)
    return I_list

def globals():
    # constants
    h = const.h # 6.62607004e-34 * J * s
    c = const.c # 299792458 * m / s

    # # Theoretical discretisation parameters
    # n = 1024
    # # # x-array parameters
    # n_x = n
    # x_max = (n_x / 2) * 5 * um
    # x = np.linspace(-x_max, x_max, n_x, endpoint=True)
    # delta_x = x[1] - x[0]

    # # Matching LAB discretisation parameters
    # Magnification
    # M = 1
    M = 2.5
    # M = 4.0

    # # x-array parameters
    delta_x = 55 * um / M
    x_max = 35 * mm / M
    x_min = -x_max
    n_x = int((x_max - x_min) / delta_x)
    print(f"\n{n_x = }")
    x = np.linspace(-x_max, x_max, n_x, endpoint=True) 

    # # Parameters from X-ray attenuation calculator   
    # # TUNGSTEN PEAKS W @ 35kV ###
    # E = 8.1 * keV # W
    # λ = h * c / E
    # k = 2 * np.pi / λ # x-rays wavenumber1
    # # Material = water, density = 1 g/cm**3
    # δ1 = 3.52955e-06
    # μ1 = 999.13349 # per m
    # β = μ1 / (2 * k)

    E = 9.7 * keV # W 
    λ = h * c / E
    k = 2 * np.pi / λ # x-rays wavenumber2
    # Material = water, density = 1 g/cm**3
    δ1 =  2.45782e-06
    μ1 = 583.22302 # per m
    β = μ1 / (2 * k)

    # E = 11.2 * keV # W
    # λ = h * c / E
    # k = 2 * np.pi / λ # x-rays wavenumber3
    # # Material = water, density = 1 g/cm**3
    # δ1 = 1.84196e-06
    # μ1 = 381.85592 # per m
    # β = μ1 / (2 * k)

    # # Bremsstrahlung radiation peak - W @ 35kV
    # E = 21 * keV 
    # λ = h * c / E
    # k = 2 * np.pi / λ # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ1 = 5.22809e-07
    # μ1 = 72.52674 # per m
    # β = μ1 / (2 * k)

    # Cylinder parameters
    D = 4 * mm
    R = D / 2

    z_c = 0 * mm
    x_c = 0 * mm

    return M, x, n_x, delta_x, E, k, δ1, μ1, β, R, z_c, x_c



# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    M, x, n_x, delta_x, E, k, δ1, μ1, β, R, z_c, x_c = globals()

    T = thicc(x, R)
    # # ICS
    Φ = - k * δ1 * T
    I_0 = np.exp(- 2 * k * β * T)

    # Φ derivatives 
    dΦ_dx, lap_Φ = gradΦ_laplacianΦ(Φ)

    # # Fourth order Runge-Kutta
    # z_final = 5 * m

    z_actual = 5 * m
    z_final = (z_actual - (z_actual / M)) / M # eff propagation distance

    delta_z = 1 * mm 
    I_list = propagation_loop(I_0) # np.shape(I_list) = (n_z / 10,  n_x)

    I = I_list[-1, :]
    np.save(f'5m_I2_2_M=2.5.npy', I)


    


