import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace
from physunits import m, cm, mm, nm, J, kg, s

plt.rcParams['figure.dpi'] = 150

# functions

def y_sigmoid(y):
    # smoothing out the edges of the cylinder in the y-direction
    𝜎_y = 0.004 * mm
    S = np.abs(1 / (1 + np.exp(-(y - height/2) / 𝜎_y)) - 
        (1 / (1 + np.exp(-(y + height/2) / 𝜎_y))))
    return S # np.shape = (n_y, 1)


def δ(x, y, z):
    '''Refractive index: δ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    δ_array = δ0 * (1 / (1 + np.exp((r - R) / 𝜎_x))) * y_sigmoid(y)
    return δ_array # np.shape(δ_array) = (n_y, n_x)


def μ(x, y, z):
    '''attenuation coefficient: μ0 within the cylinder 
    decreasing to zero at the edges Sigmoid inspired:'''
    r = np.sqrt((x - x_c) ** 2 + (z - z_c) ** 2)
    μ_array = μ0 * (1 / (1 + np.exp((r - R) / 𝜎_x))) * y_sigmoid(y)
    return μ_array # np.shape(μ_array) = (n_y, n_x)


def phase(x, y):
    # phase gain as a function of the cylinder's refractive index
    z = np.linspace(-2 * R, 2 * R, 2 ** 12, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    Φ = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        Φ += -k0 * δ(x, y, z_value) * dz
    return Φ # np.shape(Φ) = (n_y, n_x)


def BLL(x, y):
    # TIE IC of the intensity (z = z_0) a function of the cylinder's attenuation coefficient
    z = np.linspace(-2 * R, 2 * R, 2 ** 12, endpoint=False)
    dz = z[1] - z[0]
    # Euler's method
    F = np.zeros_like(x * y)
    for z_value in z:
        print(z_value)
        F += μ(x, y, z_value) * dz
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
    z_final = 1000 * mm
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
    h = 6.62607004e-34 * m**2 * kg / s
    c = 299792458 * m / s

    # x-array parameters
    n = 1024

    n_x = n
    x_max = 10 * mm
    x = np.linspace(-x_max, x_max, n_x, endpoint=False)
    delta_x = x[1] - x[0]

    # y-array parameters
    n_y = n
    y_max = 10 * mm
    y = np.linspace(-y_max, y_max, n_y, endpoint=False)#.reshape(n_y, 1)
    delta_y = y[1] - y[0]
    y = y.reshape(n_y, 1)
    
    # X-ray beam parameters
    E = 3.845e-15 * J # (Beltran et al. 2010)
    λ = h * c / E
    k0 = 2 * np.pi / λ  # x-rays wavenumber

    # # refraction and attenuation coefficients
    δ0 = 462.8 * nm # (Beltran et al. 2010)
    μ0 = 41.2 # per meter # (Beltran et al. 2010)

    # Cylinder parameters
    D = 12.75 * mm
    R = D / 2
    z_c = 0 * mm
    x_c = 0 * mm
    height = 20 * mm
    𝜎_x = 0.001 * mm

    return x, y, n_x, n_y, delta_x, delta_y, k0, R, z_c, x_c, δ0, μ0, height, 𝜎_x


# -------------------------------------------------------------------------------- #


if __name__ == '__main__':

    x, y, n_x, n_y, delta_x, delta_y, k0, R, z_c, x_c, δ0, μ0, height, 𝜎_x = globals()

    # # ICS
    Φ = phase(x, y)
    I_initial = np.ones_like(x * y)
    I_0 = BLL(x, y)

    # Φ derivatives 
    dΦ_dx, dΦ_dy, lap_Φ = gradΦ_laplacianΦ(Φ)
    # # Fourth order Runge-Kutta
    I_list = propagation_loop(I_0)

    ##################### PLOTS & TESTS #############################

    I_list = np.load("I_list.npy")  # np.shape(I_list) = (n_z / 10, n_y,  n_x)
    I = I_list[-1,:, :]
    plot_I((I)

    # # # First order finite differences
    # I_z = finite_diff(1 * m, I_0)
    # plot(I_z)