import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace
from scipy.ndimage import zoom
import scipy.constants as const
from physunits import m, cm, mm, nm, um, keV

plt.rcParams['figure.dpi'] = 150

# functions

def y_sigmoid(y):
    # smoothing out the edges of the cylinder in the y-direction
    𝜎_y = 0.004 * mm
    S = np.abs(1 / (1 + np.exp(-(y - height/2) / 𝜎_y)) - 
        (1 / (1 + np.exp(-(y + height/2) / 𝜎_y))))
    return S # np.shape = (n_y, 1)


def δ(x, y, z, δ1):
    '''Refractive index: δ1 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    δ_array = δ1 * (1 / (1 + np.exp((r - R) / 𝜎_x)))
    return δ_array # np.shape(δ_array) = (n_y, n_x)


def μ(x, y, z, μ1):
    '''attenuation coefficient: μ1 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    μ_array = μ1 * (1 / (1 + np.exp((r - R) / 𝜎_x)))
    return μ_array # np.shape(μ_array) = (n_y, n_x)


def phase(x, y):
    # phase gain as a function of the cylinder's refractive index
    z = np.linspace(-2 * R, 2 * R, 2 ** 12, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    Φ = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        Φ += -k0 * δ(x, y, z_value, δ1) * dz
    return Φ # np.shape(Φ) = (n_y, n_x)


def BLL(x, y):
    # TIE IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
    z = np.linspace(-2 * R, 2 * R, 2 ** 12, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    F = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        F += μ(x, y, z_value, μ1)* dz
    I = np.exp(- F) * I_initial
    return I # np.shape(I) = (n_y, n_x)


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
    dI_dz = (-1 / k0) * (
        dI_dx * dΦ_dx + 
        dI_dy * dΦ_dy +
        I * lap_Φ
        )
    return dI_dz  # np.shape(dI_dz) = (n_y, n_x)


def finite_diff(z, I):
    # first order finite differences
    I_z = I + z * TIE(z, I)
    return I_z


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
    z_final = 1 * m
    delta_z = 1 * mm  # (n_z = 1000)

    I = I_0
    I_list = []
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

def plot_I(I):
    # PLOT Phase contrast I in x, y
    plt.imshow(I, origin='lower')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("I")
    plt.show()

    # PLOT I vs x (a single slice)
    plt.plot(x, I[np.int(n_y / 2),:])
    plt.xlabel("x")
    plt.ylabel("I(x)")
    plt.title("Intensity profile")
    plt.show()


def globals():
    # constants
    h = const.h # 6.62607004e-34 * J * s
    c = const.c # 299792458 * m / s

    # Discretisation parameters
    # x-array parameters
    n = 1024
    n_x = n
    x_max = (n_x / 2) * 5 * um
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    delta_x = x[1] - x[0]
    # # y-array parameters
    n_y = n
    y_max = (n_y / 2) * 5 * um
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)
    delta_y = y[1] - y[0]
    y = y.reshape(n_y, 1)

    # # # X-ray beam parameters
    # # # (Beltran et al. 2010)
    # E = 3.845e-15 * J 
    # λ = h * c / E
    # # # refraction and attenuation coefficients
    # δ1 = 462.8 * nm # PMMA
    # μ1 = 41.2 # per meter # PMMA

    # # # # parameters as per energy_dispersion_Sim-1.py (MK's code)
    # energy1 = 3.5509e-15 * J #  = 22.1629 * keV #- Ag k-alpha1
    # δ1 = 468.141 * nm 
    # μ1 = 64.38436 
    # λ = h * c / energy1
    # # # secondary parameters
    # energy2 = 3.996e-15  * J # = 24.942 * keV # - Ag k-beta1
    # δ1 = 369.763 *nm
    # μ1 = 50.9387 
    # λ = h * c / energy2

    # # # TESTING HIGHER ENERGY X-RAY sample: H20 density: 1.0 g/(cm**3)
    energy1 = 50 * keV
    δ1 = 92.1425 * nm 
    μ1 = 22.69615 
    λ = h * c / energy1

    # wave number
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # Blurring 
    𝜎_x = 0.0027 * mm

    # Cylinder1 parameters
    D = 4 * mm
    R = D / 2

    z_c = 0 * mm
    x_c = 0 * mm
    height = 10 * mm

    return x, y, n_x, n_y, delta_x, delta_y, k0, R, z_c, x_c, δ1, μ1, 𝜎_x, height


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, y, n_x, n_y, delta_x, delta_y, k0, R, z_c, x_c, δ1, μ1, 𝜎_x, height = globals()

    # # ICS
    I_initial = np.ones_like(x * y)

    Φ = phase(x, y)
    I_0 = BLL(x, y)

    # Φ derivatives 
    dΦ_dx, dΦ_dy, lap_Φ = gradΦ_laplacianΦ(Φ)

    # # Fourth order Runge-Kutta
    I_list = propagation_loop(I_0) # np.shape(I_list) = (n_z / 10, n_y,  n_x)

    ##################### PLOTS & TESTS #############################

    I = I_list[-1,:, :]

    # # Re-bin step each pixel should now be 20um (for the case of 5um pixels)
    # I = zoom(I, 4.0, order=3)
    # x = zoom(x, 4.0, order=3)
    plot_I(I)

    # I_z = finite_diff(1 * m, I_0)
    # plot_I(I_z)