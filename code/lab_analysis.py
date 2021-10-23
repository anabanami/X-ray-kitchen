import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from PIL import Image

# plt.rcParams['figure.dpi'] = 300

# folder = Path('SOD62p5cm/')
folder = Path('SOD100cm/')


Flats = []
Rods = []
single_Rod = []
for p in folder.iterdir():
    if 'Flats' in p.name:
        Flats.append(p)
    if 'Rods' in p.name:
        Rods.append(p)
    else:
        single_Rod.append(p)

np_Flats_7keV = []
np_Flats_12p5keV = []
np_Flats_19keV = []
for p in Flats:
    try:
        # print(im.mode)
        np_a = np.array(Image.open(p))
    except:
        raise
    else:
        if '7keV' in p.name:
            np_Flats_7keV.append(np_a)
        elif '12p5keV' in p.name:
            np_Flats_12p5keV.append(np_a)
        else:
            np_Flats_19keV.append(np_a)

###################################################################################

# for i, array in enumerate(np_Flats_7keV):
#     print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
#     c = plt.imshow(array, vmax=3000)
#     plt.colorbar(c)
#     plt.title(f"Flatfield: 7 keV threshold, plot {i}")
#     plt.show()

# for i, array in enumerate(np_Flats_12p5keV):
#     print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
#     c = plt.imshow(array)
#     plt.colorbar(c)
#     plt.title(f"Flatfield: 12.5 keV threshold, plot {i}")
#     plt.show()

# for i, array in enumerate(np_Flats_19keV):
#     print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
#     c = plt.imshow(array)
#     plt.colorbar(c)
#     plt.title(f"Flatfield: 19 keV threshold, plot {i}")
#     plt.show()

# BOOSTING SNR
print("<<<Averaging Flats images>>>")

avg_Flats_7keV = sum(np_Flats_7keV) / len(np_Flats_7keV)
# plt.imshow(avg_Flats_7keV, vmax=4000)
# plt.colorbar()
# plt.title("Flatfield average: 7 keV threshold")
# plt.show()

avg_Flats_12p5keV = sum(np_Flats_12p5keV) / len(np_Flats_12p5keV)
# plt.imshow(avg_Flats_12p5keV, vmax=1000)
# plt.colorbar()
# plt.title("Flatfield average: 12.5 keV threshold")
# plt.show()

avg_Flats_19keV = sum(np_Flats_19keV) / len(np_Flats_19keV)
# plt.imshow(avg_Flats_19keV, vmax=250)
# plt.colorbar()
# plt.title("Flatfield average: 19 keV threshold")
# plt.show()


###################################################################################

np_Rods_7keV = []
np_Rods_12p5keV = []
np_Rods_19keV = []

for p in Rods:
    try:
        np_a = np.array(Image.open(p))
    except:
        raise
    else:
        if '7keV' in p.name:
            np_a =   ndimage.rotate(np_a / avg_Flats_7keV, -0.5)
            np_Rods_7keV.append(np_a)
        elif '12p5keV' in p.name:
            np_a = ndimage.rotate(np_a / avg_Flats_12p5keV, -0.5)
            np_Rods_12p5keV.append(np_a)
        else:
            # np_a = np_a / avg_Flats_19keV
            x = np_a / avg_Flats_19keV
            x[np.isnan(x)] = 0
            np_a = ndimage.rotate(x, -0.5)
            np_Rods_19keV.append(np_a)

print("\nTWO PERSPEX RODS (only rod2) -- IMAGES")

# for j, array in enumerate(np_Rods_7keV):
#     # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
#     plt.imshow(array[:,130:600])#, vmax=1.2)
#     plt.colorbar()
#     plt.title(f"Rods, 7 keV threshold, plot {j}")
#     plt.show()

# for j, array in enumerate(np_Rods_12p5keV):
#     # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
#     plt.imshow(array[:,130:600])#, vmax=1.5)
#     plt.colorbar()
#     plt.title(f"Rods, 12.5 keV threshold, plot {j}")
#     plt.show()

# for j, array in enumerate(np_Rods_19keV):
#     # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
#     plt.imshow(array[:,130:600])#, vmax=1.8)
#     plt.colorbar()
#     plt.title(f"Rods, 19 keV threshold, plot {j}")
#     plt.show()

# BOOSTING SNR
print("<<<Averaging images>>>")

avg_Rods_7keV = sum(np_Rods_7keV) / len(np_Rods_7keV)
if folder == Path('SOD62p5cm/'):
    avg_Rods_7keV = avg_Rods_7keV[:,75:805] # ONLY SOD62p5cm
else:
    avg_Rods_7keV = avg_Rods_7keV[:,130:600] # ONLY SOD100cm
# plt.imshow(avg_Rods_7keV)
# plt.colorbar()
# plt.title("RODS average: 7 keV threshold")
# plt.show()

avg_Rods_12p5keV = sum(np_Rods_12p5keV) / len(np_Rods_12p5keV)
if folder == Path('SOD62p5cm/'):
    avg_Rods_12p5keV = avg_Rods_12p5keV[:,75:805] # ONLY SOD62p5cm
else:
    avg_Rods_12p5keV = avg_Rods_12p5keV[:,130:600] # ONLY SOD100cm
# plt.imshow(avg_Rods_12p5keV)
# plt.colorbar()
# plt.title("RODS average: 12.5 keV threshold")
# plt.show()

avg_Rods_19keV = sum(np_Rods_19keV) / len(np_Rods_19keV)
if folder == Path('SOD62p5cm/'):
    avg_Rods_19keV = avg_Rods_19keV[:,75:805] # ONLY SOD62p5cm
else:
    avg_Rods_19keV = avg_Rods_19keV[:,130:600] # ONLY SOD100cm
# plt.imshow(avg_Rods_19keV)
# plt.colorbar()
# plt.title("RODS average: 19 keV threshold")
# plt.show()


# print("TWO PERSPEX RODS (only rod2) -- Intensity profiles")

phase_contrast_1D_7keV = np.mean(avg_Rods_7keV, axis=0)
plt.figure(figsize=(4, 3))
plt.plot(phase_contrast_1D_7keV, label="7 keV")
plt.title(f"Intensity profile ROD 2: 7 keV threshold, {folder}")
plt.legend()
plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.savefig('2_phase_contrast_1D_7keV.pdf')
plt.show()

phase_contrast_1D_12p5keV = np.mean(avg_Rods_12p5keV, axis=0)
plt.figure(figsize=(4, 3))
plt.plot(phase_contrast_1D_12p5keV, label="12.5 keV")
plt.title(f"Intensity profile ROD 2: 12.5 keV threshold, {folder}")
plt.legend()
plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.savefig('2_phase_contrast_1D_12p5keV.pdf')
plt.show()

phase_contrast_1D_19keV = np.mean(avg_Rods_19keV, axis=0)
plt.figure(figsize=(4, 3))
plt.plot(phase_contrast_1D_19keV, label="19 keV")
plt.title(f"Intensity profile ROD 2: 19 keV threshold, {folder}")
plt.legend()
plt.grid(color='grey', linestyle=':', linewidth=0.4)
# plt.savefig('2_phase_contrast_1D_19keV.pdf')
plt.show()



print("TWO PERSPEX RODS (only rod2)-- Finding the differences")

diff_1 = phase_contrast_1D_7keV - phase_contrast_1D_19keV
plt.figure(figsize=(4, 3))
plt.plot(diff_1, label="7 keV - 19 keV")
plt.title(f"Intensity profile ROD 2: differences, {folder}")
plt.legend()
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.show()


diff_2 = phase_contrast_1D_7keV - phase_contrast_1D_12p5keV
plt.figure(figsize=(4, 3))
plt.plot(diff_2, label="7 keV - 12.5 keV")
plt.title(f"Intensity profile ROD 2: differences, {folder}")
plt.legend()
plt.grid(color='grey', linestyle=':', linewidth=0.4)
plt.show()


if folder == Path('SOD62p5cm/'):

    print("\nONE PERSPEX ROD -- IMAGES")

    np_Rod_7keV = []
    np_Rod_12p5keV = []
    np_Rod_19keV = []

    for p in single_Rod:
        try:
            np_a = np.array(Image.open(p))
        except:
            raise
        else:
            if '7keV' in p.name:
                np_a = ndimage.rotate(np_a / avg_Flats_7keV, -0.5)
                np_Rod_7keV.append(np_a)
            elif '12p5keV' in p.name:
                np_a = ndimage.rotate(np_a / avg_Flats_12p5keV, -0.5)
                np_Rod_12p5keV.append(np_a)
            else:
                x = np_a / avg_Flats_19keV
                x[np.isnan(x)] = 0
                np_a = ndimage.rotate(x, -0.5)
                np_Rod_19keV.append(np_a)


    # for j, array in enumerate(np_Rod_7keV):
    #     # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
    #     plt.imshow(array)#, vmax=1.2)
    #     plt.colorbar()
    #     plt.title(f"Rod, 7 keV threshold, plot {j}")
    #     plt.show()

    # for j, array in enumerate(np_Rod_12p5keV):
    #     # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
    #     plt.imshow(array)#, vmax=1.5)
    #     plt.colorbar()
    #     plt.title(f"Rod, 12.5 keV threshold, plot {j}")
    #     plt.show()

    # for j, array in enumerate(np_Rod_19keV):
    #     # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
    #     plt.imshow(array)#, vmax=1.8)
    #     plt.colorbar()
    #     plt.title(f"Rod, 19 keV threshold, plot {j}")
    #     plt.show()


    # BOOSTING SNR
    print("<<<Averaging images>>>")

    avg_Rod_7keV = sum(np_Rod_7keV) / len(np_Rod_7keV)
    avg_Rod_7keV = avg_Rod_7keV[:,70:810] # ONLY SOD62p5
    # plt.imshow(avg_Rod_7keV)
    # plt.colorbar()
    # plt.title("ROD average: 7 keV threshold")
    # plt.show()

    avg_Rod_12p5keV = sum(np_Rod_12p5keV) / len(np_Rod_12p5keV)
    avg_Rod_12p5keV = avg_Rod_12p5keV[:,70:810] # ONLY SOD62p5
    # plt.imshow(avg_Rod_12p5keV)
    # plt.colorbar()
    # plt.title("ROD average: 12.5 keV threshold")
    # plt.show()

    avg_Rod_19keV = sum(np_Rod_19keV) / len(np_Rod_19keV)
    avg_Rod_19keV = avg_Rod_19keV[:,70:810] # ONLY SOD62p5
    # plt.imshow(avg_Rod_19keV)
    # plt.colorbar()
    # plt.title("ROD average: 19 keV threshold")
    # plt.show()

    print("ONE PERSPEX ROD -- Intensity profiles")

    phase_contrast_1D_7keV = np.mean(avg_Rod_7keV, axis=0)
    plt.figure(figsize=(4, 3))
    plt.plot(phase_contrast_1D_7keV, label="7 keV")
    plt.title(f"Intensity profile ROD 2: 7 keV threshold, {folder}")
    plt.grid(color='grey', linestyle=':', linewidth=0.4)  
    plt.show()


    phase_contrast_1D_12p5keV = np.mean(avg_Rod_12p5keV, axis=0)
    plt.figure(figsize=(4, 3))
    plt.plot(phase_contrast_1D_12p5keV, label="12.5 keV")
    plt.title(f"Intensity profile ROD 2: 12.5 keV threshold, {folder}")
    plt.grid(color='grey', linestyle=':', linewidth=0.4)
    plt.show()
    

    phase_contrast_1D_19keV = np.mean(avg_Rod_19keV, axis=0)
    plt.figure(figsize=(4, 3))
    plt.plot(phase_contrast_1D_19keV, label="19 keV")
    plt.title(f"Intensity profile ROD 2: 19 keV threshold, {folder}")
    plt.grid(color='grey', linestyle=':', linewidth=0.4)
    plt.show()



    print("ONE PERSPEX ROD -- Finding the differences")

    diff_1 = phase_contrast_1D_7keV - phase_contrast_1D_19keV
    plt.figure(figsize=(4, 3))
    plt.plot(diff_1, label="7 keV - 19 keV")
    plt.title(f"Intensity profile ROD 2: differences, {folder}")
    plt.grid(color='grey', linestyle=':', linewidth=0.4)
    plt.legend()
    plt.show()



    diff_2 = phase_contrast_1D_7keV - phase_contrast_1D_12p5keV
    plt.figure(figsize=(4, 3))
    plt.plot(diff_2, label="7 keV - 12.5 keV")
    plt.title(f"Intensity profile ROD 2: differences, {folder}")
    plt.legend()
    plt.grid(color='grey', linestyle=':', linewidth=0.4)
    plt.show()
