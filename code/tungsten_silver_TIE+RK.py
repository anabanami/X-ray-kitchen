import numpy as np
from scipy.ndimage.filters import laplace
import scipy.constants as const
from physunits import m, cm, mm, nm, um, keV
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

plt.rcParams['figure.dpi'] = 150

def thicc(x, y, R):
    # # Create cylindrical object projected thickness
    T = np.zeros_like(x * y)
    T[0:20,0:100] = 1
    T = 2 * np.sqrt(R ** 2 - x ** 2)
    T = np.nan_to_num(T)
    T = gaussian_filter(T, sigma=4)
    ones_y=np.ones_like(y)
    # # Expand 1D to 2D with outer product
    T = np.outer(ones_y, T)
    # im = plt.imshow(T)
    return T


def gradΦ_laplacianΦ(Φ):
    dΦ_dx = np.gradient(Φ, delta_x, axis=1)
    dΦ_dy = np.gradient(Φ, delta_y, axis=0)
    lap_Φ = laplace(Φ / delta_x**2)
    return dΦ_dx, dΦ_dy, lap_Φ


def TIE(z, I):
    '''The intensity and phase evolution of a paraxial monochromatic
    scalar electromagnetic wave on propagation (2D)'''
    dI_dx = np.gradient(I, delta_x, axis=1)
    dI_dy = np.gradient(I, delta_y, axis=0)
    dI_dz = (-1 / k) * (
        dI_dx * dΦ_dx + 
        dI_dy * dΦ_dy +
        I * lap_Φ
        )
    return dI_dz  # np.shape(dI_dz) = (n_y, n_x)


def Runge_Kutta(z, delta_z, I):
    # spatial evolution 4th order RK
    # z is single value, delta_z is step
    k1 = TIE(z, I)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2)
    k4 = TIE(z + delta_z, I + k3 * delta_z)    
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # shape = (n_y, n_x)


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
    print(f"{np.shape(I_list) = }") #  np.shape(I_list) = (n_z / 10, n_y, n_x)
    # np.save(f'I_list.npy', I_list)
    return I_list


def globals():
    # constants
    h = const.h # 6.62607004e-34 * J * s
    c = const.c # 299792458 * m / s

    # # Discretisation parameters
    # n = 512
    # # # x-array parameters
    # n_x = n * 2
    # x_max = (n_x / 2) * 5 * um
    # x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    # delta_x = x[1] - x[0]
    # # # y-array parameters
    # n_y = n
    # y_max = (n_y / 2) * 5 * um
    # y = np.linspace(-y_max, y_max, n_y, endpoint=False)
    # delta_y = y[1] - y[0]
    # y = y.reshape(n_y, 1)

    # # Matching LAB
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
    x = np.linspace(-x_max, x_max, n_x, endpoint=False) 
    # # y-array parameters
    delta_y = 55 * um / M
    y_max = 7 * mm / M
    y_min = -y_max
    n_y = int((y_max - y_min) / delta_y)
    print(f"\n{n_y = }")
    y = np.linspace(-y_max, y_max, n_y, endpoint=False).reshape(n_y, 1)

    # # ## Parameters from X-ray attenuation calculator   
    # ### TUNGSTEN PEAKS ###
    # E = 8.1 * keV # W
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber1
    # # Material = water, density = 1 g/cm**3
    # δ1 = 3.52955e-06
    # μ1 = 999.13349 # per m
    # β = μ1 / (2 * k)

    # E = 9.7 * keV # W 
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber2
    # # Material = water, density = 1 g/cm**3
    # δ1 =  2.45782e-06
    # μ1 = 583.22302 # per m
    # β = μ1 / (2 * k)

    # E = 11.2 * keV # W
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber3
    # # Material = water, density = 1 g/cm**3
    # δ1 = 1.84196e-06
    # μ1 = 381.85592 # per m
    # β = μ1 / (2 * k)

    # ## SILVER PEAKS ##
    # E = 21.99 * keV
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ  # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ1 = 4.76753e-07
    # μ1 = 65.62992 # per m
    # β = μ1 / (2 * k)

    # E = 24.911 * keV
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ  # x-rays wavenumber
    # # Material = water, density = 1 g/cm**3
    # δ1 = 3.71427e-07
    # μ1 = 51.15927 # per m
    # β = μ1 / (2 * k)

    # # HIGHER ENERGY ###
    # E = 19 * keV
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber2
    # # Material = water, density = 1 g/cm**3
    # δ1 =  6.38803e-07 
    # μ1 =  91.32598 # per m
    # β = μ1 / (2 * k)

    # E = 25 * keV 
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber2
    # # Material = water, density = 1 g/cm**3
    # δ1 =  3.68785e-07
    # μ1 =  50.82067 # per m
    # β = μ1 / (2 * k)

    # E = 30 * keV 
    # λ = h * c / (E * 1000 * const.eV)
    # k = 2 * np.pi / λ # x-rays wavenumber2
    # # Material = water, density = 1 g/cm**3
    # δ1 =  2.56043e-07
    # μ1 =  37.55906 # per m
    # β = μ1 / (2 * k)

    E = 35 * keV 
    λ = h * c / (E * 1000 * const.eV)
    k = 2 * np.pi / λ # x-rays wavenumber2
    # Material = water, density = 1 g/cm**3
    δ1 =  1.88086e-07
    μ1 =  30.74816 # per m
    β = μ1 / (2 * k)

    # Cylinder parameters
    D = 4 * mm
    R = D / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 10 * mm

    return M, x, y, n_x, n_y, delta_x, delta_y, E, k, δ1, μ1, β, R, z_c, x_c, height



# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    M, x, y, n_x, n_y, delta_x, delta_y, E, k, δ1, μ1, β, R, z_c, x_c, height = globals()

    T = thicc(x, y, R)
    # # ICS
    Φ = - k * δ1 * T
    # plt.plot(Φ[-1,:])
    # plt.xlabel("x")
    # plt.ylabel(R"$\phi$(x)")
    # plt.show()

    I_0 = np.exp(- 2* k * β * T)
    # plt.plot(I_0[-1,:])
    # plt.xlabel("x")
    # plt.ylabel(R"$I_{0}$(x)")
    # plt.show()

    # Φ derivatives 
    dΦ_dx, dΦ_dy, lap_Φ = gradΦ_laplacianΦ(Φ)

    # # Fourth order Runge-Kutta
    z_final = 5 * m / M # eff propagation distance
    delta_z = 1 * mm 
    I_list = propagation_loop(I_0) # np.shape(I_list) = (n_z / 10, n_y,  n_x)

    I = I_list[-1,:, :]
    np.save(f'5m_I2_9_M=2.5.npy', I)

    # I = np.load("5m_I2_1.npy")
    # I = zoom(I, 0.125, order=3)

    plt.plot(I[-10,:])
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profile")
    plt.show()


    


