import numpy as np
import math
import matplotlib.pyplot as  plt
import pandas as pd
import os

PIXEL_SIZE = 4.4e-3
PIXELS_X = 1628
PIXELS_Y = 1236
MAGNIFICATION = 0.0478
x_coord = 0
y_coord = 0
dpi = 100

x_size = PIXELS_X * PIXEL_SIZE / MAGNIFICATION
y_size = PIXELS_Y * PIXEL_SIZE / MAGNIFICATION

working_directory = os.getcwd() + "\\Group_1_cleaned\\"
folder = {
    "alpha0 SP_mean_std": "\\alpha0_SubOverTimeMin_sL=all_01_PIV_SP(32x32_50ov)_Avg_Stdev=unknown\\",
    "alpha5 SP_mean_std": "\\alpha5_SubOverTimeMin_sL=all_PIV_SP(32x32_50ov)_Avg_Stdev=unknown\\",
    "alpha15 SP_mean_std": "\\alpha15_SubOverTimeMin_sL=all_PIV_SP(32x32_50ov)_Avg_Stdev=unknown\\",
    "alpha15 instant": "\\alpha15_SubOverTimeMin_sL=all_PIV_SP(32x32_50ov)=unknown_01\\",
    "alpha15 quick_SP_mean_std": "\\AoA_15_sdtdeg_SubOverTimeMin_sL=5_PIV_SP(32x32_50ov)_Avg_Stdev=unknown\\"
          }

# Create array and function, so all filenames can be generated
number = [str(n).zfill(2) for n in range(1, 51)]
file = lambda number: f"B000{number}.dat"

cmap = plt.get_cmap('jet')
cmap.set_under("black")  # Color for values less than vmin

def calculate_mean_velocity(file_path):
    with open(file_path, "r") as file:
        file_data = file.readlines()


    column_names = ["x", "y", "vx", "vy", "falid"]
    rows = [line.split() for line in file_data[3:]]
    data = np.array(rows, dtype=float)

    # Remove row if last column is not falid and create spatial and velocity arrays
    x = data[:, 0]
    y = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]

    global x_coord
    global y_coord

    x_coord = np.unique(x)
    y_coord = np.unique(y)

    X, Y = np.meshgrid(x_coord, y_coord)
    mean_velocity = np.sqrt(vx**2 + vy**2)
    velocity = mean_velocity.reshape([len(y_coord), len(x_coord)])

    return mean_velocity, velocity
#%% INSTANTENEOUS
mean_velocity, velocity = calculate_mean_velocity(working_directory + \
                                                  folder["alpha15 instant"] + \
                                                  file(number[0]))

#velocity = mean_velocity.reshape([len(y_coord), len(x_coord)])

plt.figure(figsize=(12, 8), dpi=dpi)
plt.title(r"Single pass 32x32 @ 50 % OL. Instantaneous at $15 \degree$.")
image = plt.imshow(velocity,
                    vmin = mean_velocity[mean_velocity > 0].min(),
                    vmax = max(mean_velocity),
                    cmap = cmap,
                    extent = [0, x_size, 0, y_size])
colorbar = plt.colorbar(image, extend="min")
colorbar.set_label('|v| [m/s]', rotation=90)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.show()

#%% INSTATENEOUS small delta t
mean_velocity, velocity = calculate_mean_velocity(working_directory + \
                                                  folder["alpha15 quick_SP_mean_std"] + \
                                                  file(number[0]))

#velocity = mean_velocity.reshape([len(y_coord), len(x_coord)])

plt.figure(figsize=(12, 8), dpi=dpi)
plt.title(r"Single pass 32x32 @50 % overlap. Mean at $15 \degree$. Small dt")
image = plt.imshow(velocity,
                    vmin = mean_velocity[mean_velocity > 0].min(),
                    vmax = max(mean_velocity),
                    cmap = cmap,
                    extent = [0, x_size, 0, y_size])
colorbar = plt.colorbar(image, extend="min")
colorbar.set_label('|v| [m/s]', rotation=90)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.show()

#%% AVG and STD plots
for degree in [0, 5, 15]:

    for i in number[:2]:
        mean_velocity, velocity = calculate_mean_velocity(working_directory + \
                                                      folder[f"alpha{degree} SP_mean_std"] + \
                                                      file(i))
        #velocity = mean_velocity.reshape([len(y_coord), len(x_coord)])

        if i == "01":
            name = "Mean"
        elif i == "02":
            name = "Standard deviation"
        else:
            raise "File does not exist."

        plt.figure(figsize=(12, 8), dpi=dpi)
        plt.title(fr"Single pass 32x32 50 % overlap. {name} at {degree}$\degree$.")
        image = plt.imshow(velocity,
                            vmin = mean_velocity[mean_velocity > 0].min(),
                            vmax = max(mean_velocity),
                            cmap = cmap,
                            extent = [0, x_size, 0, y_size])
        colorbar = plt.colorbar(image, extend="min")
        if i =="01":
            colorbar.set_label('|v| [m/s]', rotation=90)
        else:
            colorbar.set_label('std_Dev []', rotation=90)
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.show()


#%% Ensemble averages
sample_size = [5, 20, 50]


for sample in sample_size:
    ensemble_mean_velocity = np.zeros((sample, len(mean_velocity)))
    for i in range(sample):
        ensemble_mean_velocity[i, :], velocity = calculate_mean_velocity(working_directory + \
                                                      folder[f"alpha15 instant"] + \
                                                      file(number[i]))
    ensemble = ensemble_mean_velocity.sum(axis=0)
    plt.figure(figsize=(12, 8), dpi=dpi)
    plt.title(fr"Single pass 32x32 50 % overlap. Mean at 15$\degree$, ensamble average of {sample}.")
    image = plt.imshow(ensemble.reshape([len(y_coord), len(x_coord)]),
                        vmin = ensemble[ensemble > 0].min(),
                        vmax = max(ensemble),
                        cmap = cmap,
                        extent = [0, x_size, 0, y_size])
    colorbar = plt.colorbar(image, extend="min")
    colorbar.set_label('|v| [m/s]', rotation=90)
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.show()

#%% coord length
coordlenght = 100 # [mm]
x_over_c = 1.2
x0 = 20
x_pos_tail = x0 + coordlenght * x_over_c

line_x_mean_velocity = velocity[:, int(len(x_coord) * x_pos_tail / x_size)]
plt.figure(figsize=(12, 8), dpi=dpi)
plt.title(f"Velocity profile at x/c=1.2")
plt.plot(line_x_mean_velocity, y_coord)
plt.xlabel("velocity [m/s]")
plt.ylabel("y [mm]")
plt.show()
