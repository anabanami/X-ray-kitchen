import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from PIL import Image

plt.rcParams['figure.dpi'] = 300

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
print("<<<Averaging images>>>")

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

corrected_np_Rods_7keV = []
corrected_np_Rods_12p5keV = []
corrected_np_Rods_19keV = []

for p in Rods:
    try:
        np_a = np.array(Image.open(p))
    except:
        raise
    else:
        if '7keV' in p.name:
            corrected_np_a =   ndimage.rotate(np_a / avg_Flats_7keV, -0.39)
            corrected_np_Rods_7keV.append(corrected_np_a)
        elif '12p5keV' in p.name:
            corrected_np_a = ndimage.rotate(np_a / avg_Flats_12p5keV, -0.39)
            corrected_np_Rods_12p5keV.append(corrected_np_a)
        else:
            # corrected_np_a = np_a / avg_Flats_19keV
            corrected_np_a = ndimage.rotate(np_a / avg_Flats_19keV, -0.39)
            corrected_np_Rods_19keV.append(corrected_np_a)

print("\nTWO PERSPEX RODS -- IMAGES")

for j, array in enumerate(corrected_np_Rods_7keV):
    # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
    plt.imshow(array[:,130:600])#, vmax=1.2)
    plt.colorbar()
    plt.title(f"Corrected Rods, 7 keV threshold, plot {j}")
    plt.show()

for j, array in enumerate(corrected_np_Rods_12p5keV):
    # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
    plt.imshow(array[:,130:600])#, vmax=1.5)
    plt.colorbar()
    plt.title(f"Corrected Rods, 12.5 keV threshold, plot {j}")
    plt.show()

for j, array in enumerate(corrected_np_Rods_19keV):
    # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
    plt.imshow(array[:,130:600])#, vmax=1.8)
    plt.colorbar()
    plt.title(f"Corrected Rods, 19 keV threshold, plot {j}")
    plt.show()

# BOOSTING SNR
print("<<<Averaging images>>>")

avg_Rods_7keV = sum(corrected_np_Rods_7keV) / len(corrected_np_Rods_7keV)
# plt.imshow(avg_Rods_7keV)
# plt.colorbar()
# plt.title("RODS average: 7 keV threshold")
# plt.show()

avg_Rods_12p5keV = sum(corrected_np_Rods_12p5keV) / len(corrected_np_Rods_12p5keV)
# plt.imshow(avg_Rods_12p5keV)
# plt.colorbar()
# plt.title("RODS average: 12.5 keV threshold")
# plt.show()

avg_Rods_19keV = sum(corrected_np_Rods_19keV) / len(corrected_np_Rods_19keV)
# # plt.imshow(avg_Rods_19keV)
# # plt.colorbar()
# # plt.title("RODS average: 19 keV threshold")
# # plt.show()

print("TWO PERSPEX RODS -- Intensity profiles")

phase_contrast_1D_7keV = []
for row in avg_Rods_7keV:
    row_average = np.mean(row)
    phase_contrast_1D_7keV.append(row_average)

phase_contrast_1D_7keV = np.array(phase_contrast_1D_7keV)
plt.plot(phase_contrast_1D_7keV, label="7 keV")
plt.title("Intensity profile RODS: 7 keV keV threshold")
plt.show()

phase_contrast_1D_12p5keV = []
for row in avg_Rods_12p5keV:
    row_average = np.mean(row)
    phase_contrast_1D_12p5keV.append(row_average)

phase_contrast_1D_12p5keV = np.array(phase_contrast_1D_12p5keV)
plt.plot(phase_contrast_1D_12p5keV, label="12.5 keV")
plt.title("Intensity profile RODS: 12.5 keV keV threshold")
plt.show()

phase_contrast_1D_19keV = []
for row in avg_Rods_19keV:
    row_average = np.mean(row)
    phase_contrast_1D_19keV.append(row_average)

phase_contrast_1D_19keV = np.array(phase_contrast_1D_19keV)
plt.plot(phase_contrast_1D_19keV, label="19 keV")
plt.title("Intensity profile RODS: 19 keV keV threshold")
plt.show()

if folder == Path('SOD62p5cm/'):

    print("\nONE PERSPEX ROD -- IMAGES")

    corrected_np_Rod_7keV = []
    corrected_np_Rod_12p5keV = []
    corrected_np_Rod_19keV = []

    for p in single_Rod:
        try:
            np_a = np.array(Image.open(p))
        except:
            raise
        else:
            if '7keV' in p.name:
                corrected_np_a = np_a / avg_Flats_7keV
                corrected_np_Rod_7keV.append(corrected_np_a)
            elif '12p5keV' in p.name:
                corrected_np_a = np_a / avg_Flats_12p5keV
                corrected_np_Rod_12p5keV.append(corrected_np_a)
            else:
                corrected_np_a = np_a / avg_Flats_19keV
                corrected_np_Rod_19keV.append(corrected_np_a)


    for j, array in enumerate(corrected_np_Rod_7keV):
        # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
        c = plt.imshow(array)#, vmax=1.2)
        plt.colorbar(c)
        plt.title(f"Corrected Rod, 7 keV threshold, plot {j}")
        plt.show()

    for j, array in enumerate(corrected_np_Rod_12p5keV):
        # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
        c = plt.imshow(array)#, vmax=1.5)
        plt.colorbar(c)
        plt.title(f"Corrected Rod, 12.5 keV threshold, plot {j}")
        plt.show()

    for j, array in enumerate(corrected_np_Rod_19keV):
        # print(f"{array.min()}, {array.max()}, {array.mean()}, {array.std()}")
        c = plt.imshow(array)#, vmax=1.8)
        plt.colorbar(c)
        plt.title(f"Corrected Rod, 19 keV threshold, plot {j}")
        plt.show()


    # # BOOSTING SNR
    # print("<<<Averaging images>>>")

    # avg_Rod_7keV = sum(corrected_np_Rod_7keV) / len(corrected_np_Rod_7keV)
    # # plt.imshow(avg_Rods_7keV, vmax=4000)
    # # plt.colorbar()
    # # plt.title("RODS average: 7 keV threshold")
    # # plt.show()

    # avg_Rod_12p5keV = sum(corrected_np_Rod_12p5keV) / len(corrected_np_Rod_12p5keV)
    # # plt.imshow(avg_Rods_12p5keV, vmax=1000)
    # # plt.colorbar()
    # # plt.title("RODS average: 12.5 keV threshold")
    # # plt.show()

    # avg_Rod_19keV = sum(corrected_np_Rod_19keV) / len(corrected_np_Rod_19keV)
    # # plt.imshow(avg_Rods_19keV, vmax=250)
    # # plt.colorbar()
    # # plt.title("RODS average: 19 keV threshold")
    # # plt.show()

    # print("ONE PERSPEX ROD -- Intensity profiles")

    # phase_contrast_1D_7keV = []
    # for row in avg_Rod_7keV:
    #     row_average = np.mean(row)
    #     phase_contrast_1D_7keV.append(row_average)

    # phase_contrast_1D_7keV = np.array(phase_contrast_1D_7keV)
    # plt.plot(phase_contrast_1D_7keV, label="7 keV")
    # plt.title("Intensity profile ROD: 7 keV keV threshold")
    # plt.show()

    # phase_contrast_1D_12p5keV = []
    # for row in avg_Rod_12p5keV:
    #     row_average = np.mean(row)
    #     phase_contrast_1D_12p5keV.append(row_average)

    # phase_contrast_1D_12p5keV = np.array(phase_contrast_1D_12p5keV)
    # plt.plot(phase_contrast_1D_12p5keV, label="12.5 keV")
    # plt.title("Intensity profile ROD: 12.5 keV keV threshold")
    # plt.show()

    # phase_contrast_1D_19keV = []
    # for row in avg_Rod_19keV:
    #     row_average = np.mean(row)
    #     phase_contrast_1D_19keV.append(row_average)

    # phase_contrast_1D_19keV = np.array(phase_contrast_1D_19keV)
    # plt.plot(phase_contrast_1D_19keV, label="19 keV")
    # plt.title("Intensity profile ROD: 19 keV keV threshold")
    # plt.show()