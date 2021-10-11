import numpy as np
import matplotlib
import matplotlib.pyplot as plt
    
plt.rcParams['figure.dpi'] = 200


# # # LOAD: LAB parameters AS 
# _1m_1 = np.load("1m_I1_1.npy")
# _1m_1 = _1m_1[-100,:]
# _1m_2 = np.load("1m_I1_2.npy")
# _1m_2 = _1m_2[-100,:]
# _1m_3 = np.load("1m_I1_3.npy")   
# _1m_3 = _1m_3[-100,:]
# _1m_4 = np.load("1m_I1_4.npy")
# _1m_4 = _1m_4[-100,:]
# _2_5m_1 = np.load("2.5m_I1_1.npy")
# _2_5m_1 = _2_5m_1[-100,:]
# _2_5m_2 = np.load("2.5m_I1_2.npy")
# _2_5m_2 = _2_5m_2[-100,:]
# _2_5m_3 = np.load("2.5m_I1_3.npy")  
# _2_5m_3 = _2_5m_3[-100,:]
# _2_5m_4 = np.load("2.5m_I1_4.npy")
# _2_5m_4 = _2_5m_4[-100,:]
# _5m_1 = np.load("5m_I1_1.npy")
# _5m_1 = _5m_1[-100,:]
# _5m_2 = np.load("5m_I1_2.npy")
# _5m_2 = _5m_2[-100,:]
# _5m_3 = np.load("5m_I1_3.npy")
# _5m_3 = _5m_3[-100,:]
# _5m_4 = np.load("5m_I1_4.npy")
# _5m_4 = _5m_4[-100,:]

# # # LOAD: LAB parameters TIE + RK
# _1m_1 = np.load("1m_I2_1.npy")
# _1m_2 = np.load("1m_I2_2.npy")
# _1m_3 = np.load("1m_I2_3.npy")   
# _1m_4 = np.load("1m_I2_4.npy")
# _2_5m_1 = np.load("2.5m_I2_1.npy")
# _2_5m_2 = np.load("2.5m_I2_2.npy")
# _2_5m_3 = np.load("2.5m_I2_3.npy")  
# _2_5m_4 = np.load("2.5m_I2_4.npy")
# _5m_1 = np.load("5m_I2_1.npy")
# _5m_2 = np.load("5m_I2_2.npy")
# _5m_3 = np.load("5m_I2_3.npy")
# _5m_4 = np.load("5m_I2_4.npy")

# # LOAD: LAB parameters TIE
_1m_1 = np.load("1m_I3_1.npy")
_1m_1 = _1m_1[-100,:]
_1m_2 = np.load("1m_I3_2.npy")
_1m_2 = _1m_2[-100,:]
_1m_3 = np.load("1m_I3_3.npy")   
_1m_3 = _1m_3[-100,:]
_1m_4 = np.load("1m_I3_4.npy")
_1m_4 = _1m_4[-100,:]
_2_5m_1 = np.load("2.5m_I3_1.npy")
_2_5m_1 = _2_5m_1[-100,:]
_2_5m_2 = np.load("2.5m_I3_2.npy")
_2_5m_2 = _2_5m_2[-100,:]
_2_5m_3 = np.load("2.5m_I3_3.npy")  
_2_5m_3 = _2_5m_3[-100,:]
_2_5m_4 = np.load("2.5m_I3_4.npy")
_2_5m_4 = _2_5m_4[-100,:]
_5m_1 = np.load("5m_I3_1.npy")
_5m_1 = _5m_1[-100,:]
_5m_2 = np.load("5m_I3_2.npy")
_5m_2 = _5m_2[-100,:]
_5m_3 = np.load("5m_I3_3.npy")
_5m_3 = _5m_3[-100,:]
_5m_4 = np.load("5m_I3_4.npy")
_5m_4 = _5m_4[-100,:]

# # average between energies
average_1 = 1/4 * (_1m_1 + _1m_2 + _1m_3 + _1m_4)
average_2_5 = 1/4 * (_2_5m_1 + _2_5m_2 + _2_5m_3 + _2_5m_4)
average_5 = 1/4 * (_5m_1 + _5m_2 + _5m_3 + _5m_4)

# # ## Downsample! ###
# _1m_1 = _1m_1.reshape(int(len(_1m_1 ) / 4), 4).mean(axis = -1)
# _1m_2 = _1m_2.reshape(int(len(_1m_2 ) / 4), 4).mean(axis = -1)
# _1m_3 = _1m_3.reshape(int(len(_1m_3 ) / 4), 4).mean(axis = -1)
# _1m_4 = _1m_4.reshape(int(len(_1m_4 ) / 4), 4).mean(axis = -1)

# _2_5m_1 = _2_5m_1.reshape(int(len( _2_5m_1 ) / 4), 4).mean(axis = -1)
# _2_5m_2 = _2_5m_2.reshape(int(len( _2_5m_2 ) / 4), 4).mean(axis = -1)
# _2_5m_3 = _2_5m_3.reshape(int(len( _2_5m_3 ) / 4), 4).mean(axis = -1)
# _2_5m_4 = _2_5m_4.reshape(int(len(_2_5m_4 ) / 4), 4).mean(axis = -1)

# _5m_1 = _5m_1.reshape(int(len(_5m_1 ) / 4), 4).mean(axis = -1)
# _5m_2 = _5m_2.reshape(int(len(_5m_2 ) / 4), 4).mean(axis = -1)
# _5m_3 = _5m_3.reshape(int(len(_5m_3 ) / 4), 4).mean(axis = -1)
# _5m_4 = _5m_4.reshape(int(len(_5m_4 ) / 4), 4).mean(axis = -1)

# # # downsampled average between energies
# average_1 = average_1.reshape(int(len(average_1) / 4), 4).mean(axis = -1)
# average_2_5 = average_2_5.reshape(int(len(average_2_5) / 4), 4).mean(axis = -1)
# average_5 = average_5.reshape(int(len(average_5) / 4), 4).mean(axis = -1)


# #######################################################################################

# # DIFFERENT ENERGY PEAKS @ SAME PROP DISTANCES ###
# # 1m
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
# # plt.title(fR"AS: Tungsten peaks z = 1 m")
# plt.title(fR"TIE+RK: Tungsten peaks z = 1 m")
# # plt.title(fR"TIE: Tungsten peaks  z = 1 m")
# plt.show()

# 2.5m
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
# plt.title(fR"AS: Tungsten peaks z = 2.5 m")
# plt.title(fR"TIE+RK: Tungsten peaks  z = 2.5 m")
plt.title(fR"TIE: Tungsten peaks  z = 2.5 m")
plt.show()

# # # 5m
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
# # plt.title(fR"AS: Tungsten peaks z = 5 m")
# plt.title(fR"TIE+RK: Tungsten peaks  z = 5 m")
# # plt.title(fR"TIE: Tungsten peaks  z = 5 m")
# plt.show()


########################################################################################

## SAME ENERGY PEAKS @ DIFFERENT PROP DISTANCES ###
plt.plot(_1m_1, label=fR"$z_{{eff}}$ = 1 m")
plt.plot(_2_5m_1, label=fR"$z_{{eff}}$ = 2.5 m")
plt.plot(_5m_1, label=fR"$z_{{eff}}$ = 5 m")
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 8.1 keV")
# plt.title("TIE+RK: Intensity profiles: W at 8.1 keV")
plt.title("TIE: Intensity profiles: W at 8.1 keV")
plt.legend()
plt.show()

plt.plot(_1m_2, label=fR"$z_{{eff}}$ = 1 m")
plt.plot(_2_5m_2, label=fR"$z_{{eff}}$ = 2.5 m")
plt.plot(_5m_2, label=fR"$z_{{eff}}$ = 5 m")
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 9.7 keV")
# plt.title("TIE+RK: Intensity profiles: W at 9.7 keV")
plt.title("TIE: Intensity profiles: W at 9.7 keV")
plt.legend()
plt.show()

plt.plot(_1m_3, label=fR"$z_{{eff}}$ = 1 m")
plt.plot(_2_5m_3, label=fR"$z_{{eff}}$ = 2.5 m")
plt.plot(_5m_3, label=fR"$z_{{eff}}$ = 5 m")
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 11.2 keV")
# plt.title("TIE+RK: Intensity profiles: W at 11.2 keV")
plt.title("TIE: Intensity profiles: W at 11.2 keV")
plt.legend()
plt.show()

plt.plot(_1m_4, label=fR"$z_{{eff}}$ = 1 m")
plt.plot(_2_5m_4, label=fR"$z_{{eff}}$ = 2.5 m")
plt.plot(_5m_4, label=fR"$z_{{eff}}$ = 5 m")
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.xlabel("x")
plt.ylabel("I(x)")
# plt.title("AS: Intensity profiles: W at 21 keV")
# plt.title("TIE+RK: Intensity profiles: W at 21 keV")
plt.title("TIE: Intensity profiles: W at 21 keV")
plt.legend()
plt.show()



# #######################################################################################

# # AS vs TIE + RK vs TIE ###

# _1_2_5m_1 = np.load("2.5m_I1_1.npy")
# _1_2_5m_1 = _1_2_5m_1[-100,:]
# _2_2_5m_1 = np.load("2.5m_I2_1.npy")
# _3_2_5m_1 = np.load("2.5m_I3_1.npy")
# _3_2_5m_1 = _3_2_5m_1[-100,:]

# # # # downsample
# # _1_2_5m_1 = _1_2_5m_1.reshape(int(len( _1_2_5m_1 ) / 4), 4).mean(axis = -1)
# # _2_2_5m_1 = _2_2_5m_1.reshape(int(len( _2_2_5m_1 ) / 4), 4).mean(axis = -1)
# # _3_2_5m_1 = _3_2_5m_1.reshape(int(len( _3_2_5m_1 ) / 4), 4).mean(axis = -1)

# _1_2_5m_2 = np.load("2.5m_I1_2.npy")
# _1_2_5m_2 = _1_2_5m_2[-100,:]
# _2_2_5m_2 = np.load("2.5m_I2_2.npy")
# _3_2_5m_2 = np.load("2.5m_I3_2.npy")
# _3_2_5m_2 = _3_2_5m_2[-100,:]

# # # # downsample
# # _1_2_5m_2 = _1_2_5m_2.reshape(int(len( _1_2_5m_2 ) / 4), 4).mean(axis = -1)
# # _2_2_5m_2 = _2_2_5m_2.reshape(int(len( _2_2_5m_2 ) / 4), 4).mean(axis = -1)
# # _3_2_5m_2 = _3_2_5m_2.reshape(int(len( _3_2_5m_2 ) / 4), 4).mean(axis = -1)

# _1_2_5m_3 = np.load("2.5m_I1_3.npy")
# _1_2_5m_3 = _1_2_5m_3[-100,:]
# _2_2_5m_3 = np.load("2.5m_I2_3.npy")
# _3_2_5m_3 = np.load("2.5m_I3_3.npy")
# _3_2_5m_3 = _3_2_5m_3[-100,:]

# # # # downsample
# # _1_2_5m_3 = _1_2_5m_3.reshape(int(len( _1_2_5m_3 ) / 4), 4).mean(axis = -1)
# # _2_2_5m_3 = _2_2_5m_3.reshape(int(len( _2_2_5m_3 ) / 4), 4).mean(axis = -1)
# # _3_2_5m_3 = _3_2_5m_3.reshape(int(len( _3_2_5m_3 ) / 4), 4).mean(axis = -1)

# _1_2_5m_4 = np.load("2.5m_I1_4.npy")
# _1_2_5m_4 = _1_2_5m_4[-100,:]
# _2_2_5m_4 = np.load("2.5m_I2_4.npy")
# _3_2_5m_4 = np.load("2.5m_I3_4.npy")
# _3_2_5m_4 = _3_2_5m_4[-100,:]

# # # # downsample
# # _1_2_5m_4 = _1_2_5m_4.reshape(int(len( _1_2_5m_4 ) / 4), 4).mean(axis = -1)
# # _2_2_5m_4 = _2_2_5m_4.reshape(int(len( _2_2_5m_4 ) / 4), 4).mean(axis = -1)
# # _3_2_5m_4 = _3_2_5m_4.reshape(int(len( _3_2_5m_4 ) / 4), 4).mean(axis = -1)

# #plots
# plt.plot(_1_2_5m_1, label="AS")
# plt.plot(_2_2_5m_1, label="TIE+RK")
# plt.plot(_3_2_5m_1, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Characteristic X-ray peak $8.1$ keV at z = 2.5 m")
# plt.show()

# plt.plot(_1_2_5m_2, label="AS")
# plt.plot(_2_2_5m_2, label="TIE+RK")
# plt.plot(_3_2_5m_2, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Characteristic X-ray peak $9.7$ keV at z = 2.5 m")
# plt.show()

# plt.plot(_1_2_5m_3, label="AS")
# plt.plot(_2_2_5m_3, label="TIE+RK")
# plt.plot(_3_2_5m_3, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Characteristic X-ray peak $11.2$ keV at z = 2.5 m")
# plt.show()

# plt.plot(_1_2_5m_4, label="AS")
# plt.plot(_2_2_5m_4, label="TIE+RK")
# plt.plot(_3_2_5m_4, label="xri's TIE")
# plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("I(x)")
# plt.title(fR"Tungsten 35kV: Bremsstrahlung peak $21$ keV at z = 2.5 m")
# plt.show()
