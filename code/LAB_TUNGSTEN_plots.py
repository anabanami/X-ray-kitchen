import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import special
import scipy.optimize
    
plt.rcParams['figure.dpi'] = 200

# # # LOAD: LAB parameters AS @ M = 2.5X
# M = 2.5
# _1m_1 = np.load("1m_I1_1_M=2.5.npy")
# _1m_1 = _1m_1[-100,:]
# _1m_2 = np.load("1m_I1_2_M=2.5.npy")
# _1m_2 = _1m_2[-100,:]
# _1m_3 = np.load("1m_I1_3_M=2.5.npy")   
# _1m_3 = _1m_3[-100,:]
# _1m_4 = np.load("1m_I1_4_M=2.5.npy")
# _1m_4 = _1m_4[-100,:]
# _2_5m_1 = np.load("2.5m_I1_1_M=2.5.npy")
# _2_5m_1 = _2_5m_1[-100,:]
# _2_5m_2 = np.load("2.5m_I1_2_M=2.5.npy")
# _2_5m_2 = _2_5m_2[-100,:]
# _2_5m_3 = np.load("2.5m_I1_3_M=2.5.npy")  
# _2_5m_3 = _2_5m_3[-100,:]
# _2_5m_4 = np.load("2.5m_I1_4_M=2.5.npy")
# _2_5m_4 = _2_5m_4[-100,:]
# _5m_1 = np.load("5m_I1_1_M=2.5.npy")
# _5m_1 = _5m_1[-100,:]
# _5m_2 = np.load("5m_I1_2_M=2.5.npy")
# _5m_2 = _5m_2[-100,:]
# _5m_3 = np.load("5m_I1_3_M=2.5.npy")
# _5m_3 = _5m_3[-100,:]
# _5m_4 = np.load("5m_I1_4_M=2.5.npy")
# _5m_4 = _5m_4[-100,:]

# # # LOAD: LAB parameters AS @ M = 4.0X
# M = 4.0
# _1m_1 = np.load("1m_I1_1_M=4.npy")
# _1m_1 = _1m_1[-100,:]
# _1m_2 = np.load("1m_I1_2_M=4.npy")
# _1m_2 = _1m_2[-100,:]
# _1m_3 = np.load("1m_I1_3_M=4.npy")   
# _1m_3 = _1m_3[-100,:]
# _1m_4 = np.load("1m_I1_4_M=4.npy")
# _1m_4 = _1m_4[-100,:]
# _2_5m_1 = np.load("2.5m_I1_1_M=4.npy")
# _2_5m_1 = _2_5m_1[-100,:]
# _2_5m_2 = np.load("2.5m_I1_2_M=4.npy")
# _2_5m_2 = _2_5m_2[-100,:]
# _2_5m_3 = np.load("2.5m_I1_3_M=4.npy")  
# _2_5m_3 = _2_5m_3[-100,:]
# _2_5m_4 = np.load("2.5m_I1_4_M=4.npy")
# _2_5m_4 = _2_5m_4[-100,:]
# _5m_1 = np.load("5m_I1_1_M=4.npy")
# _5m_1 = _5m_1[-100,:]
# _5m_2 = np.load("5m_I1_2_M=4.npy")
# _5m_2 = _5m_2[-100,:]
# _5m_3 = np.load("5m_I1_3_M=4.npy")
# _5m_3 = _5m_3[-100,:]
# _5m_4 = np.load("5m_I1_4_M=4.npy")
# _5m_4 = _5m_4[-100,:]


# ## LOAD: LAB parameters TIE + RK @ M = 2.5X
# M = 2.5
# _1m_1 = np.load("1m_I2_1_M=2.5.npy")
# _1m_2 = np.load("1m_I2_2_M=2.5.npy")
# _1m_3 = np.load("1m_I2_3_M=2.5.npy")   
# _1m_4 = np.load("1m_I2_4_M=2.5.npy")
# _2_5m_1 = np.load("2.5m_I2_1_M=2.5.npy")
# _2_5m_2 = np.load("2.5m_I2_2_M=2.5.npy")
# _2_5m_3 = np.load("2.5m_I2_3_M=2.5.npy")    
# _2_5m_4 = np.load("2.5m_I2_4_M=2.5.npy")
# _5m_1 = np.load("5m_I2_1_M=2.5.npy")
# _5m_2 = np.load("5m_I2_2_M=2.5.npy")
# _5m_3 = np.load("5m_I2_3_M=2.5.npy")  
# _5m_4 = np.load("5m_I2_4_M=2.5.npy")

# # # LOAD: LAB parameters TIE + RK @ M = 4.0X
# M = 4.0
# _1m_1 = np.load("1m_I2_1_M=4.npy")
# _1m_2 = np.load("1m_I2_2_M=4.npy")
# _1m_3 = np.load("1m_I2_3_M=4.npy")   
# _1m_4 = np.load("1m_I2_4_M=4.npy")
# _2_5m_1 = np.load("2.5m_I2_1_M=4.npy")
# _2_5m_2 = np.load("2.5m_I2_2_M=4.npy")
# _2_5m_3 = np.load("2.5m_I2_3_M=4.npy")  
# _2_5m_4 = np.load("2.5m_I2_4_M=4.npy")
# _5m_1 = np.load("5m_I2_1_M=4.npy")
# _5m_2 = np.load("5m_I2_2_M=4.npy")
# _5m_3 = np.load("5m_I2_3_M=4.npy")
# _5m_4 = np.load("5m_I2_4_M=4.npy")


# # LOAD: LAB parameters TIE @ M = 2.5X
# M = 2.5
# _1m_1 = np.load("1m_I3_1_M=2.5.npy")
# _1m_1 = _1m_1[-100,:]
# _1m_2 = np.load("1m_I3_2_M=2.5.npy")
# _1m_2 = _1m_2[-100,:]
# _1m_3 = np.load("1m_I3_3_M=2.5.npy")   
# _1m_3 = _1m_3[-100,:]
# _1m_4 = np.load("1m_I3_4_M=2.5.npy")
# _1m_4 = _1m_4[-100,:]
# _2_5m_1 = np.load("2.5m_I3_1_M=2.5.npy")
# _2_5m_1 = _2_5m_1[-100,:]
# _2_5m_2 = np.load("2.5m_I3_2_M=2.5.npy")
# _2_5m_2 = _2_5m_2[-100,:]
# _2_5m_3 = np.load("2.5m_I3_3_M=2.5.npy")  
# _2_5m_3 = _2_5m_3[-100,:]
# _2_5m_4 = np.load("2.5m_I3_4_M=2.5.npy")
# _2_5m_4 = _2_5m_4[-100,:]
# _5m_1 = np.load("5m_I3_1_M=2.5.npy")
# _5m_1 = _5m_1[-100,:]
# _5m_2 = np.load("5m_I3_2_M=2.5.npy")
# _5m_2 = _5m_2[-100,:]
# _5m_3 = np.load("5m_I3_3_M=2.5.npy")
# _5m_3 = _5m_3[-100,:]
# _5m_4 = np.load("5m_I3_4_M=2.5.npy")
# _5m_4 = _5m_4[-100,:]

# # LOAD: LAB parameters TIE @ M = 4.0X
M = 4.0
_1m_1 = np.load("1m_I3_1_M=4.npy")
_1m_1 = _1m_1[-100,:]
_1m_2 = np.load("1m_I3_2_M=4.npy")
_1m_2 = _1m_2[-100,:]
_1m_3 = np.load("1m_I3_3_M=4.npy")   
_1m_3 = _1m_3[-100,:]
_1m_4 = np.load("1m_I3_4_M=4.npy")
_1m_4 = _1m_4[-100,:]
_2_5m_1 = np.load("2.5m_I3_1_M=4.npy")
_2_5m_1 = _2_5m_1[-100,:]
_2_5m_2 = np.load("2.5m_I3_2_M=4.npy")
_2_5m_2 = _2_5m_2[-100,:]
_2_5m_3 = np.load("2.5m_I3_3_M=4.npy")  
_2_5m_3 = _2_5m_3[-100,:]
_2_5m_4 = np.load("2.5m_I3_4_M=4.npy")
_2_5m_4 = _2_5m_4[-100,:]
_5m_1 = np.load("5m_I3_1_M=4.npy")
_5m_1 = _5m_1[-100,:]
_5m_2 = np.load("5m_I3_2_M=4.npy")
_5m_2 = _5m_2[-100,:]
_5m_3 = np.load("5m_I3_3_M=4.npy")
_5m_3 = _5m_3[-100,:]
_5m_4 = np.load("5m_I3_4_M=4.npy")
_5m_4 = _5m_4[-100,:]

# # average between energies
average_1 = 1/4 * (_1m_1 + _1m_2 + _1m_3 + _1m_4)
average_2_5 = 1/4 * (_2_5m_1 + _2_5m_2 + _2_5m_3 + _2_5m_4)
average_5 = 1/4 * (_5m_1 + _5m_2 + _5m_3 + _5m_4)

# ## Downsample! ###
_1m_1 = _1m_1.reshape(int(len(_1m_1 ) / 4), 4).mean(axis = -1)
_1m_2 = _1m_2.reshape(int(len(_1m_2 ) / 4), 4).mean(axis = -1)
_1m_3 = _1m_3.reshape(int(len(_1m_3 ) / 4), 4).mean(axis = -1)
_1m_4 = _1m_4.reshape(int(len(_1m_4 ) / 4), 4).mean(axis = -1)

_2_5m_1 = _2_5m_1.reshape(int(len( _2_5m_1 ) / 4), 4).mean(axis = -1)
_2_5m_2 = _2_5m_2.reshape(int(len( _2_5m_2 ) / 4), 4).mean(axis = -1)
_2_5m_3 = _2_5m_3.reshape(int(len( _2_5m_3 ) / 4), 4).mean(axis = -1)
_2_5m_4 = _2_5m_4.reshape(int(len(_2_5m_4 ) / 4), 4).mean(axis = -1)

_5m_1 = _5m_1.reshape(int(len(_5m_1 ) / 4), 4).mean(axis = -1)
_5m_2 = _5m_2.reshape(int(len(_5m_2 ) / 4), 4).mean(axis = -1)
_5m_3 = _5m_3.reshape(int(len(_5m_3 ) / 4), 4).mean(axis = -1)
_5m_4 = _5m_4.reshape(int(len(_5m_4 ) / 4), 4).mean(axis = -1)

# # downsampled average between energies
average_1 = average_1.reshape(int(len(average_1) / 4), 4).mean(axis = -1)
average_2_5 = average_2_5.reshape(int(len(average_2_5) / 4), 4).mean(axis = -1)
average_5 = average_5.reshape(int(len(average_5) / 4), 4).mean(axis = -1)


########################################################################################

# # DIFFERENT ENERGY PEAKS @ SAME PROP DISTANCES ###
# # 1m
# SID = 1
# SOD = SID / M
# plt.plot(_1m_1, label="E = 8.1 keV")
# plt.plot(_1m_2, label="E = 9.7 keV")
# plt.plot(_1m_3, label="E = 11.2 keV")
# plt.plot(_1m_4, label="E = 21 keV")
# plt.plot(average_1, linestyle="--", label="average")
# # plt.axvline(121, color='pink', linestyle="--", linewidth=0.7, label="fringe")
# # plt.axvline(196, color='pink', linestyle="--", linewidth=0.7)
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"AS: Tungsten peaks $z_{{eff}} = {(SID - SOD)/ M} m$")
# # plt.title(fR"TIE+RK: Tungsten peaks $z_{{eff}} = {(SID - SOD)/ M} m$")
# # plt.title(fR"TIE: Tungsten peaks $z_{{eff}} = {(SID - SOD)/ M} m$")
# plt.show()

# 2.5m
SID = 2.5
SOD = SID / M
plt.plot(_2_5m_1, label="E = 8.1 keV")
plt.plot(_2_5m_2, label="E = 9.7 keV")
plt.plot(_2_5m_3, label="E = 11.2 keV")
plt.plot(_2_5m_4, label="E = 21 keV")
plt.plot(average_2_5, linestyle="--", label="average")
# plt.axvline(121, color='pink', linestyle="--", linewidth=0.7, label="fringe")
# plt.axvline(196, color='pink', linestyle="--", linewidth=0.7)
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.legend()
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title(fR"AS: Tungsten peaks $z_{{eff}} = {(SID - SOD)/ M} m$")
# plt.title(fR"TIE+RK: Tungsten peaks $z_{{eff}}= {(SID - SOD) / M} m$")
plt.title(fR"TIE: Tungsten peaks $z_{{eff}}= {(SID - SOD) / M} m$")
plt.show()

# # # 5m
# SID = 5
# SOD = SID / M
# plt.plot(_5m_1, label="E = 8.1 keV")
# plt.plot(_5m_2, label="E = 9.7 keV")
# plt.plot(_5m_3, label="E = 11.2 keV")
# plt.plot(_5m_4, label="E = 21 keV")
# plt.plot(average_5, linestyle="--", label="average")
# # plt.axvline(121, color='pink', linestyle="--", linewidth=0.7, label="fringe")
# # plt.axvline(196, color='pink', linestyle="--", linewidth=0.7)
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"AS: Tungsten peaks $z_{{eff}} = {(SID - SOD)/ M} m$")
# # plt.title(fR"TIE+RK: Tungsten peaks $z_{{eff}} = {(SID - SOD) / M} m$")
# # plt.title(fR"TIE: Tungsten peaks $z_{{eff}} = {(SID - SOD) / M} m$")
# plt.show()


########################################################################################

## SAME ENERGY PEAKS @ DIFFERENT PROP DISTANCES ###
plt.plot(_1m_1, label=fR"$z_{{eff}}$ = {(1 - (1/M)) / M} m")
plt.plot(_2_5m_1, label=fR"$z_{{eff}}$ = {(2.5 - (2.5/M)) / M} m")
plt.plot(_5m_1, label=fR"$z_{{eff}}$ = {(5 - (5/M)) / M} m")
# plt.axvline(121, color='purple', linestyle="--", linewidth=0.7, label=" fringe")
# plt.axvline(196, color='purple', linestyle="--", linewidth=0.7)
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 8.1 keV")
# plt.title("TIE+RK: Intensity profiles: W at 8.1 keV")
plt.title("TIE: Intensity profiles: W at 8.1 keV")
plt.legend()
plt.show()

plt.plot(_1m_2, label=fR"$z_{{eff}}$ = {(1 - (1/M)) / M} m")
plt.plot(_2_5m_2, label=fR"$z_{{eff}}$ = {(2.5 - (2.5/M)) / M} m")
plt.plot(_5m_2, label=fR"$z_{{eff}}$ = {(5 - (5/M)) / M} m")
# plt.axvline(121, color='purple', linestyle="--", linewidth=0.7, label=" fringe")
# plt.axvline(196, color='purple', linestyle="--", linewidth=0.7)
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 9.7 keV")
# plt.title("TIE+RK: Intensity profiles: W at 9.7 keV")
plt.title("TIE: Intensity profiles: W at 9.7 keV")
plt.legend()
plt.show()

plt.plot(_1m_3, label=fR"$z_{{eff}}$ = {(1 - (1/M)) / M} m")
plt.plot(_2_5m_3, label=fR"$z_{{eff}}$ = {(2.5 - (2.5/M)) / M} m")
plt.plot(_5m_3, label=fR"$z_{{eff}}$ = {(5 - (5/M)) / M} m")
# plt.axvline(121, color='purple', linestyle="--",linewidth=0.5, label="fringe")
# plt.axvline(196, color='purple', linestyle="--", linewidth=0.7)
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 11.2 keV")
# plt.title("TIE+RK: Intensity profiles: W at 11.2 keV")
plt.title("TIE: Intensity profiles: W at 11.2 keV")
plt.legend()
plt.show()

plt.plot(_1m_4, label=fR"$z_{{eff}}$ = {(1 - (1/M)) / M} m")
plt.plot(_2_5m_4, label=fR"$z_{{eff}}$ = {(2.5 - (2.5/M)) / M} m")
plt.plot(_5m_4, label=fR"$z_{{eff}}$ = {(5 - (5/M)) / M} m")
# plt.axvline(121, color='purple', linestyle="--", linewidth=0.7, label=" fringe")
# plt.axvline(196, color='purple', linestyle="--", linewidth=0.7)
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 21 keV")
# plt.title("TIE+RK: Intensity profiles: W at 21 keV")
plt.title("TIE: Intensity profiles: W at 21 keV")
plt.legend()
plt.show()


# ########################################################################################

# ## AS vs TIE + RK vs TIE ###

# M = 2.5
# _1_2_5m_1 = np.load("2.5m_I1_1_M=2.5.npy") # AS
# _1_2_5m_1 = _1_2_5m_1[-100,:]
# _2_2_5m_1 = np.load("2.5m_I2_1_M=2.5.npy") # TIE + RK
# _3_2_5m_1 = np.load("2.5m_I3_1_M=2.5.npy") # TIE
# _3_2_5m_1 = _3_2_5m_1[-100,:]

# _1_2_5m_2 = np.load("2.5m_I1_2_M=2.5.npy") # AS
# _1_2_5m_2 = _1_2_5m_2[-100,:]
# _2_2_5m_2 = np.load("2.5m_I2_2_M=2.5.npy") # TIE + RK
# _3_2_5m_2 = np.load("2.5m_I3_2_M=2.5.npy") # TIE
# _3_2_5m_2 = _3_2_5m_2[-100,:]

# _1_2_5m_3 = np.load("2.5m_I1_3_M=2.5.npy") # AS
# _1_2_5m_3 = _1_2_5m_3[-100,:]
# _2_2_5m_3 = np.load("2.5m_I2_3_M=2.5.npy") # TIE + RK
# _3_2_5m_3 = np.load("2.5m_I3_3_M=2.5.npy") # TIE
# _3_2_5m_3 = _3_2_5m_3[-100,:]

# _1_2_5m_4 = np.load("2.5m_I1_4_M=2.5.npy") # AS
# _1_2_5m_4 = _1_2_5m_4[-100,:]
# _2_2_5m_4 = np.load("2.5m_I2_4_M=2.5.npy") # TIE + RK
# _3_2_5m_4 = np.load("2.5m_I3_4_M=2.5.npy") # TIE
# _3_2_5m_4 = _3_2_5m_4[-100,:]

# #####################################################

# M = 4.0
# _1_2_5m_1 = np.load("2.5m_I1_1_M=4.npy") # AS
# _1_2_5m_1 = _1_2_5m_1[-100,:]
# _2_2_5m_1 = np.load("2.5m_I2_1_M=4.npy") # TIE + RK
# _3_2_5m_1 = np.load("2.5m_I3_1_M=4.npy") # TIE
# _3_2_5m_1 = _3_2_5m_1[-100,:]

# _1_2_5m_2 = np.load("2.5m_I1_2_M=4.npy") # AS
# _1_2_5m_2 = _1_2_5m_2[-100,:]
# _2_2_5m_2 = np.load("2.5m_I2_2_M=4.npy") # TIE + RK
# _3_2_5m_2 = np.load("2.5m_I3_2_M=4.npy") # TIE
# _3_2_5m_2 = _3_2_5m_2[-100,:]

# _1_2_5m_3 = np.load("2.5m_I1_3_M=4.npy") # AS
# _1_2_5m_3 = _1_2_5m_3[-100,:]
# _2_2_5m_3 = np.load("2.5m_I2_3_M=4.npy") # TIE + RK
# _3_2_5m_3 = np.load("2.5m_I3_3_M=4.npy") # TIE
# _3_2_5m_3 = _3_2_5m_3[-100,:]

# _1_2_5m_4 = np.load("2.5m_I1_4_M=4.npy") # AS
# _1_2_5m_4 = _1_2_5m_4[-100,:]
# _2_2_5m_4 = np.load("2.5m_I2_4_M=4.npy") # TIE + RK
# _3_2_5m_4 = np.load("2.5m_I3_4_M=4.npy") # TIE
# _3_2_5m_4 = _3_2_5m_4[-100,:]

# # # # resample

# _1_2_5m_1 = _1_2_5m_1.reshape(int(len(_1_2_5m_1) / 4), 4).mean(axis = -1)
# _2_2_5m_1 = _2_2_5m_1.reshape(int(len(_2_2_5m_1) / 4), 4).mean(axis = -1)
# _3_2_5m_1 = _3_2_5m_1.reshape(int(len(_3_2_5m_1) / 4), 4).mean(axis = -1)

# _1_2_5m_2 = _1_2_5m_2.reshape(int(len(_1_2_5m_2) / 4), 4).mean(axis = -1)
# _2_2_5m_2 = _2_2_5m_2.reshape(int(len(_2_2_5m_2) / 4), 4).mean(axis = -1)
# _3_2_5m_2 = _3_2_5m_2.reshape(int(len(_3_2_5m_2) / 4), 4).mean(axis = -1)

# _1_2_5m_3 = _1_2_5m_3.reshape(int(len(_1_2_5m_3) / 4), 4).mean(axis = -1)
# _2_2_5m_3 = _2_2_5m_3.reshape(int(len(_2_2_5m_3) / 4), 4).mean(axis = -1)
# _3_2_5m_3 = _3_2_5m_3.reshape(int(len(_3_2_5m_3) / 4), 4).mean(axis = -1)

# _1_2_5m_4 = _1_2_5m_4.reshape(int(len(_1_2_5m_4) / 4), 4).mean(axis = -1)
# _2_2_5m_4 = _2_2_5m_4.reshape(int(len(_2_2_5m_4) / 4), 4).mean(axis = -1)
# _3_2_5m_4 = _3_2_5m_4.reshape(int(len(_3_2_5m_4) / 4), 4).mean(axis = -1)

# ### 2.5 m ###
# SID = 2.5
# SOD = SID / M

# #plot
# plt.plot(_1_2_5m_1, label="AS")
# plt.plot(_2_2_5m_1, label="TIE+RK")
# plt.plot(_3_2_5m_1, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Characteristic X-ray peak $8.1$ keV at $z_{{eff}}$ = {(SID - SOD) / M} m")
# plt.show()

# plt.plot(_1_2_5m_2, label="AS")
# plt.plot(_2_2_5m_2, label="TIE+RK")
# plt.plot(_3_2_5m_2, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Characteristic X-ray peak $9.7$ keV at $z_{{eff}}$ = {(SID - SOD) / M} m")
# plt.show()

# plt.plot(_1_2_5m_3, label="AS")
# plt.plot(_2_2_5m_3, label="TIE+RK")
# plt.plot(_3_2_5m_3, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Characteristic X-ray peak $11.2$ keV at $z_{{eff}}$ = {(SID - SOD) / M} m")
# plt.show()

# plt.plot(_1_2_5m_4, label="AS")
# plt.plot(_2_2_5m_4, label="TIE+RK")
# plt.plot(_3_2_5m_4, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Bremsstrahlung peak $21$ keV at $z_{{eff}}$ = {(SID - SOD) / M} m")
# plt.show()
