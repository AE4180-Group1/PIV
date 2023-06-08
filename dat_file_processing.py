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

for root, dirs, files in os.walk("./Group_1_cleaned/"):
    for file in files:
        if file.endswith(".dat"):
             print(os.path.join(root, file))

             create_vector_heatmap_plot(os.path.join(root, file))
        break # can be remove once code is finished
    break # can be removed once code is finished
    