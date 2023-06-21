import numpy as np
import math
import matplotlib.pyplot as plt
import os


def create_vector_heatmap_plot(path):
    with open(path, "r") as file:
        file_data = file.readlines()


    column_names = ["x", "y", "vx", "vy", "falid"]
    rows = [line.split() for line in file_data[3:]]
    data = np.array(rows, dtype=float)
    
    # Remove row if last column is not falid and create spatial and velocity arrays
    # data = data[data[:, -1] == 1, :] # Drops invalid data
    x = data[:, 0]
    y = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]
    
    # Create grid
    N_x = 4
    N_y = 3
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    print(min(x), max(x))
    print(min(y), max(y))
    gridpoints_x = np.linspace(math.floor(min(x)),
                               math.ceil(max(x)),
                               int(x_range / N_x ))
    gridpoints_y = np.linspace(math.floor(min(y)),
                               math.ceil(max(y)),
                               int(y_range / N_y))
    
    # Create velocity matrix, with x and y velocity at every gridpoint
    averaged_vx = np.zeros([len(gridpoints_y), len(gridpoints_x)])
    averaged_vy = np.zeros([len(gridpoints_y), len(gridpoints_x)])
    
    # loop over gridpoints lying in gridcell i,j
    for j in range(len(gridpoints_y) - 1):
        for i in range(len(gridpoints_x) - 1):
            indices_inside_gridcell = np.where((x > gridpoints_x[i]) & 
                                               (x <= gridpoints_x[i+1]) &
                                               (y > gridpoints_y[j]) &
                                               (y <= gridpoints_y[j+1]))
         
            averaged_vx[j, i] = vx[indices_inside_gridcell].mean()
            averaged_vy[j, i] = vy[indices_inside_gridcell].mean()
    
    
    X, Y = np.meshgrid(gridpoints_x, gridpoints_y)
    
    
    plt.figure(figsize=(12,8), dpi=200)
    plt.contourf(X, Y, averaged_vx, levels=10, cmap="jet")
    plt.colorbar()
    
    plt.title("Air around airfoil with heatmap denoting the velocity in x direction")
    plt.quiver(X, Y, averaged_vx, averaged_vy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

create_vector_heatmap_plot("C:/Users/markb/OneDrive/Cloud_Documents/TU_Delft/Master/Flow_Measurement_Techiques/project/PIV/Group_1_cleaned/alpha0_SubOverTimeMin_sL=all_01_PIV_MP(3x32x32_50ov)_Avg_Stdev=unknown/B00001.dat")


# for root, dirs, files in os.walk(os.getcwd() + "PIV/Group_1_cleaned/"):
#     print(root)
#     for file in files:
#         print(file)
#         if file.endswith(".dat"):
#              print(os.path.join(root, file))

#              create_vector_heatmap_plot(os.path.join(root, file))
#         break # can be remove once code is finished
#     break # can be removed once code is finished
#%%
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

x_size = PIXELS_X * PIXEL_SIZE / MAGNIFICATION
y_size = PIXELS_Y * PIXEL_SIZE / MAGNIFICATION

working_directory = os.getcwd() + "\\PIV\\Group_1_cleaned\\"
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
    
    return mean_velocity
#%% INSTATENEOUS
mean_velocity = calculate_mean_velocity(working_directory + \
                                                  folder["alpha15 instant"] + \
                                                  file(number[0]))
    
velocity = mean_velocity.reshape([len(y_coord), len(x_coord)])

plt.figure(figsize=(12, 8), dpi=200)
plt.title("Velocity magnitude plot of air around airfoil flowing in positive direction. \n" + \
          r"Single pass 32x32 interogation window (50\% overlap). Instatanious at $15 \degree$.")
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
mean_velocity = calculate_mean_velocity(working_directory + \
                                                  folder["alpha15 quick_SP_mean_std"] + \
                                                  file(number[0]))
    
velocity = mean_velocity.reshape([len(y_coord), len(x_coord)])

plt.figure(figsize=(12, 8), dpi=200)
plt.title("Velocity magnitude plot of air around airfoil flowing in positive direction. Small delta t.\n" + \
          r"Single pass 32x32 interogation window (50\% overlap). Instatanious at $15 \degree$.")
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
        if i == "01":
            name = "Mean"
        elif i == "02": 
            name = "Standard deviation"
        else:
            raise "File does not exist."
            
        plt.figure(figsize=(12, 8), dpi=200)
        plt.title(f"{name} of velocity magnitude plot of air around airfoil flowing in positive direction. \n" + \
                  fr"Single pass 32x32 interogation window (50\% overlap). Instatanious at {degree}$\degree$.")
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


#%% Ensemble averages
sample_size = [5, 20, 50]


for sample in sample_size:
    ensemble_mean_velocity = np.zeros((sample, len(mean_velocity)))
    for i in range(sample):
        ensemble_mean_velocity[i, :] = calculate_mean_velocity(working_directory + \
                                                      folder[f"alpha15 instant"] + \
                                                      file(number[i]))
    ensemble = ensemble_mean_velocity.sum(axis=0)
    plt.figure(figsize=(12, 8), dpi=200)
    plt.title(f"Ensemble average ({sample}) of velocity magnitude plot of air around airfoil flowing in positive direction. \n" + \
              fr"Single pass 32x32 interogation window (50\% overlap). Instatanious at 15$\degree$.")
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
plt.figure(figsize=(12, 8), dpi=200)
plt.title(f"")
plt.plot(line_x_mean_velocity, y_coord)
plt.xlabel("velocity [m/s]")
plt.ylabel("y [mm]")
plt.show()