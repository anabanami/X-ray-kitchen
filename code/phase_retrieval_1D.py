import numpy as np
from scipy.ndimage.filters import laplace
import scipy.constants as const
from physunits import m, cm, mm, nm, um, keV
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150


def gradΦ_laplacianΦ(Φ):
    dΦ_dx = np.gradient(Φ, delta_x)
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
    # z is single value, delta_z is evolution step
    k1 = TIE(z, I)
    k2 = TIE(z + delta_z / 2, I + k1 * delta_z / 2)
    k3 = TIE(z + delta_z / 2, I + k2 * delta_z / 2)
    k4 = TIE(z + delta_z, I + k3 * delta_z)    
    return I + (delta_z / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # shape = (n_x)


def back_propagation_loop(z_eff, delta_z, I_0):
    # RK Propagation loop parameters
    j = 0
    z = z_eff

    # checking dimension of intensity array
    if len(np.shape(I_0)) == 1:
        print("<<< 1D array >>>")
        I = I_0
    else:
        print("<<< 2D array >>>")
        I = I_0[-1,:]

    I_list = []
    print("<< propagating wavefield >>") 
    while z > 0:
        print(f"{j = }")
        # print(f"{z = }")

        # spatial evolution
        I = Runge_Kutta(z, -delta_z, I) # (n_x,)
        if not j % 10:
            I_list.append(I)

        if not first_iteration and not j % 10:
            plt.plot(I)
            plt.xlabel("x")
            plt.ylabel("I")
            plt.show()
        j += 1
        z -= delta_z

    I_list = np.array(I_list)
    return I_list # shape = (n_z / 10, n_x,)

def phase_retrieval(I_0):

    global Φ, dΦ_dx, lap_Φ, first_iteration

    for i in range(3):
        first_iteration = i == 0
        print(f"{i = }")

        # print(f"{np.shape(I_0) = }")
        I = back_propagation_loop(z_eff, delta_z, I_0)
        I = I[-1,:] # np.shape(I) = (n_x,)
        # print(f"{np.shape(I) = }")

        # # Step 3: obtain T from new intensity
        T = (-1 / μ) * np.log(I)# / I_in) 

        # # Step 4: use T to calculate a new phase
        Φ = - k * δ * T
        print(f"{np.shape(Φ) = }")
        # new phase derivatives for TIE
        dΦ_dx, lap_Φ = gradΦ_laplacianΦ(Φ)
        
        # Φ changes 
        plt.plot(Φ)
        plt.xlabel("x")
        plt.ylabel(R"$\phi(x)$")
        plt.title("phase profile")
        plt.show()
        
        # # I changes 
        plt.plot(I)
        plt.xlabel("x")
        plt.ylabel("I")
        plt.show()     
        
    return Φ, dΦ_dx, lap_Φ

def globals():
    # constants
    h = const.h # 6.62607004e-34 * J * s
    c = const.c # 299792458 * m / s

    # Theoretical discretisation parameters
    n = 512
    # # x-array parameters
    n_x = n * 2
    x_max = (n_x / 2) * 5 * um
    x = np.linspace(-x_max, x_max, n_x, endpoint=True)
    delta_x = x[1] - x[0]

    # # Matching LAB discretisation parameters
    # Magnification
    M = 1
    # M = 2.5
    # M = 4.0

    # # # x-array parameters
    # delta_x = 55 * um / M
    # x_max = 35 * mm / M
    # x_min = -x_max
    # n_x = int((x_max - x_min) / delta_x)
    # # print(f"\n{n_x = }")
    # x = np.linspace(-x_max, x_max, n_x, endpoint=True) 

    # refractive index and attenuation coefficient
    ## TUNGSTEN PEAKS W @ 35kV ###
    E = 8.1 * keV # W
    λ = h * c / (E * 1000 * const.eV)
    k = 2 * np.pi / λ # x-rays wavenumber1
    # Material = water, density = 1 g/cm**3
    δ = 3.52955e-06
    μ = 999.13349 # per m
    β = μ / (2 * k)

    I_in = np.ones_like(x)

    return M, x, n_x, delta_x, δ, μ, k, I_in



# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    M, x, n_x, delta_x, δ, μ, k, I_in = globals()

    # # Step 1: from phase contrast obtain I_0
    # I_0 = np.load("5m_I1_1_M=2.5.npy")
    # I_0 = np.load("5m_I2_1_M=2.5.npy")
    # I_0 = np.load("5m_I3_1_M=2.5.npy")
    I_0 = np.load("test_W.npy")


    # # # Step 1.2: guess a phase
    Φ = np.zeros_like(x)
    dΦ_dx, lap_Φ = gradΦ_laplacianΦ(Φ)

    # # Step 2: run 4th order RK towards contact plane
    z_eff = 1 * m / M # eff propagation distance
    delta_z = 1 * mm 


    Φ, dΦ_dx, lap_Φ = phase_retrieval(I_0)
    # phase = np.save(f".npy", Φ)

    # plt.plot(Φ)
    # plt.xlabel("x")
    # plt.ylabel(R"$\phi(x)$")
    # plt.title("phase profile")
    # plt.show()


