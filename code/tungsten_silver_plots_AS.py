import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from physunits import m, cm, mm, nm, um
import scipy.constants as const
import xri
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
    
plt.rcParams['figure.dpi'] = 200

### LOAD ###

_1m_1 = np.load("1m_I1_1.npy")
_1m_2 = np.load("1m_I1_2.npy")
_1m_3 = np.load("1m_I1_3.npy")    
_1m_4 = np.load("1m_I1_4.npy")
_1m_5 = np.load("1m_I1_5.npy")
_1m_6 = np.load("1m_I1_6.npy")
_1m_7 = np.load("1m_I1_7.npy")
_1m_8 = np.load("1m_I1_8.npy")
_1m_9 = np.load("1m_I1_9.npy")

_2_5m_1 = np.load("2.5m_I1_1.npy")
_2_5m_2 = np.load("2.5m_I1_2.npy")
_2_5m_3 = np.load("2.5m_I1_3.npy")    
_2_5m_4 = np.load("2.5m_I1_4.npy")
_2_5m_5 = np.load("2.5m_I1_5.npy")
_2_5m_6 = np.load("2.5m_I1_6.npy")
_2_5m_7 = np.load("2.5m_I1_7.npy")
_2_5m_8 = np.load("2.5m_I1_8.npy")
_2_5m_9 = np.load("2.5m_I1_9.npy")

_5m_1 = np.load("5m_I1_1.npy")
_5m_2 = np.load("5m_I1_2.npy")
_5m_3 = np.load("5m_I1_3.npy")    
_5m_4 = np.load("5m_I1_4.npy")
_5m_5 = np.load("5m_I1_5.npy")
_5m_6 = np.load("5m_I1_6.npy")
_5m_7 = np.load("5m_I1_7.npy")
_5m_8 = np.load("5m_I1_8.npy")
_5m_9 = np.load("5m_I1_9.npy")

### ENHANCE! ###

# _1m_1 = zoom(_1m_1, 0.5, order=3)
# _1m_2 = zoom(_1m_2, 0.5, order=3)
# _1m_3 = zoom(_1m_3, 0.5, order=3)
# _1m_4 = zoom(_1m_4, 0.5, order=3)
# _1m_5 = zoom(_1m_5, 0.5, order=3)
# _1m_6 = zoom(_1m_6, 0.5, order=3)
# _1m_7 = zoom(_1m_7, 0.5, order=3)
# _1m_8 = zoom(_1m_8, 0.5, order=3)
# _1m_9 = zoom(_1m_9, 0.5, order=3)

# _2_5m_1 = zoom(_2_5m_1, 0.5, order=3)
# _2_5m_2 = zoom(_2_5m_2, 0.5, order=3)
# _2_5m_3 = zoom(_2_5m_3, 0.5, order=3)
# _2_5m_4 = zoom(_2_5m_4, 0.5, order=3)
# _2_5m_5 = zoom(_2_5m_5, 0.5, order=3)
# _2_5m_6 = zoom(_2_5m_6, 0.5, order=3)
# _2_5m_7 = zoom(_2_5m_7, 0.5, order=3)
# _2_5m_8 = zoom(_2_5m_8, 0.5, order=3)
# _2_5m_9 = zoom(_2_5m_9, 0.5, order=3)

# _5m_1 = zoom(_5m_1, 0.5, order=3)
# _5m_2 = zoom(_5m_2, 0.5, order=3)
# _5m_3 = zoom(_5m_3, 0.5, order=3)
# _5m_4 = zoom(_5m_4, 0.5, order=3)
# _5m_5 = zoom(_5m_5, 0.5, order=3)
# _5m_6 = zoom(_5m_6, 0.5, order=3)
# _5m_7 = zoom(_5m_7, 0.5, order=3)
# _5m_8 = zoom(_5m_8, 0.5, order=3)
# _5m_9 = zoom(_5m_9, 0.5, order=3)

### DIFFERENT ENERGY PEAKS @ SAME PROP DISTANCES ###

plt.plot(_1m_1[-1], label="E = 8.1 keV")
plt.plot(_1m_2[-1], label="E = 9.7 keV")
plt.plot(_1m_3[-1], label="E = 11.2 keV")
plt.plot(_1m_6[-1], label="E = 19 keV")
plt.plot(_1m_7[-1], label="E = 25 keV")
plt.axvline(104, color="pink", ls=":", label=r"W peak at $z_{\mathrm{eff}}$ = 1 m")
plt.axvline(920, color="pink", ls=":")
plt.legend()
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title(r"Tungsten peaks $z_{eff} = 1 m$")
plt.show()

plt.plot(_1m_4[-1], label="E = 21.99 keV")
plt.plot(_1m_5[-1], label="E = 24.911 keV")
plt.plot(_1m_8[-1], label="E = 30 keV")
plt.plot(_1m_9[-1], label="E = 35 keV")
plt.axvline(109, color="grey", ls=":", label=r"Ag peak at $z_{\mathrm{eff}}$ = 1 m")
plt.axvline(915, color="grey", ls=":")
plt.legend()
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title("Silver peaks $z_{eff} = 1 m$")
plt.show()

### SAME ENERGY PEAKS @ DIFFERENT PROP DISTANCES ###

plt.plot(_1m_1[-1], label=r"$z_{\mathrm{eff}}$ = 1 m")
plt.plot(_2_5m_1[-1], label=r"$z_{\mathrm{eff}}$ = 2.5 m")
plt.plot(_5m_1[-1], label=r"$z_{\mathrm{eff}}$ = 5 m")
# plt.axvline(104, color="pink", ls=":", label=r"W peak at $z_{\mathrm{eff}}$ = 1 m")
# plt.axvline(920, color="pink", ls=":")
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title("Intensity profiles W at 8.1 keV")
plt.legend()
plt.show()

plt.plot(_1m_2[-1], label=r"$z_{\mathrm{eff}}$ = 1 m")
plt.plot(_2_5m_2[-1], label=r"$z_{\mathrm{eff}}$ = 2.5 m")
plt.plot(_5m_2[-1], label=r"$z_{\mathrm{eff}}$ = 5 m")
# plt.axvline(104, color="pink", ls=":", label=r"W peak at $z_{\mathrm{eff}}$ = 1 m")
# plt.axvline(920, color="pink", ls=":")
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title("Intensity profiles W at 9.7 keV")
plt.legend()
plt.show()

plt.plot(_1m_3[-1], label=r"$z_{\mathrm{eff}}$ = 1 m")
plt.plot(_2_5m_3[-1], label=r"$z_{\mathrm{eff}}$ = 2.5 m")
plt.plot(_5m_3[-1], label=r"$z_{\mathrm{eff}}$ = 5 m")
# plt.axvline(104, color="pink", ls=":", label=r"W peak at $z_{\mathrm{eff}}$ = 1 m")
# plt.axvline(920, color="pink", ls=":")
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title("Intensity profiles W at 11.2 keV")
plt.legend()
plt.show()

plt.plot(_1m_4[-1], label=r"$z_{\mathrm{eff}}$ = 1 m")
plt.plot(_2_5m_4[-1], label=r"$z_{\mathrm{eff}}$ = 2.5 m")
plt.plot(_5m_4[-1], label=r"$z_{\mathrm{eff}}$ = 5 m")
# plt.axvline(109, color="grey", ls=":", label=r"Ag peak at $z_{\mathrm{eff}}$ = 1 m")
# plt.axvline(915, color="grey", ls=":")
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title("Intensity profiles Ag at 21.99 keV")
plt.legend()
plt.show()

plt.plot(_1m_5[-1], label=r"$z_{\mathrm{eff}}$ = 1 m")
plt.plot(_2_5m_5[-1], label=r"$z_{\mathrm{eff}}$ = 2.5 m")
plt.plot(_5m_5[-1], label=r"$z_{\mathrm{eff}}$ = 5 m")
# plt.axvline(109, color="grey", ls=":", label=r"Ag peak at $z_{\mathrm{eff}}$ = 1 m")
# plt.axvline(915, color="grey", ls=":")
plt.xlabel("x")
plt.ylabel("I(x)")
plt.title("Intensity profiles Ag at 24.911 keV")
plt.legend()
plt.show()