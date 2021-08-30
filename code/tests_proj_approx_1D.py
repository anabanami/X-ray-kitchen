    # I = BLL(x)
    # dI_dz = TIE(z, I, Φ) 

    # # dI_dz Test plot
    # plt.plot(x, dI_dz)
    # plt.xlabel("x")
    # plt.ylabel("dI_dz")
    # plt.title(r"TIE: $\frac{\partial I(x)}{\partial z}$ ")
    # plt.show()

    # # I Test plot
    # plt.plot(x, I)
    # plt.xlabel("x")
    # plt.ylabel("I")
    # plt.title(r"Beer-Lamber law: $I(x)$ ")
    # plt.show()

    # # PLOT ATTENUATION FACTOR I/I0 vs x after RK
    # plt.plot(x, I_list[1000] / I_0)
    # plt.xlabel("x")
    # plt.ylabel(r"$I(x)/I_{0}$")
    # plt.title(r"Attenuation factor: $I(x)/I_{0}$ ")
    # plt.show()

    # # PLOT Φ vs x
    # plt.plot(x, Φ)
    # plt.xlabel("x")
    # plt.ylabel(r"$\phi(x)$")
    # plt.title(r"Phase shift $\phi(x) = -k_{0} \int^{z_{0}}_{0} \delta(x, z) dz$ ")
    # plt.show()

    # test_list = np.arange(n_x)
    # print(f"\n{np.shape(test_list) = }")
    # print(f"\n{test_list = }")

    # test_list = np.reshape(test_list, (n_x, 1))
    # print(f"\n{np.shape(test_list) = }")
    # print(f"\n{test_list = }")
    