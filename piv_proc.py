import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import interpolate
from scipy import optimize
import math

# Defining the variables
AoA = 15           # airfoil angle of attack (0, 5 or 15)
mask_on = 1          # 0 - mask off; 1 - mask on
Int_Win = 32      # Interrogation window size in pixles
Overlap = 50          # Overlap of interrogation windows in %
dt = 72*10**-6    # Time step between two frames

# Calibration
p1 = [300.8, 317.0]
p2 = [1236.3, 335.9]
pix_dist = math.dist(p1, p2)
pix_vel = 0.11/(pix_dist*dt) # speed per 1px of movement

#Loading images
img_test = cv.imread("Group_1_cleaned/AoA_15_sdtdeg_SubOverTimeMin_sL=5/B00005.tif", -1)
#img_src = cv.imread("Group_1_cleaned/alpha0_SubOverTimeMin_sL=all_01/B00019.tif", -1) #0
#img_src = cv.imread("Group_1_cleaned/alpha5_SubOverTimeMin_sL=all/B00019.tif", -1) #5
#img_src = cv.imread("Group_1_cleaned/alpha15_SubOverTimeMin_sL=all/B00019.tif", -1) #15
img_src = cv.imread("images/Alpha15_Diff/B00002.tif", -1)
img_src_cal = cv.imread("images/Calibration_02/B00001.tif", -1)[1236:, :]
mask0 = cv.imread("images/mask0.tif")
mask5 = cv.imread("images/mask5.tif")
mask15 = cv.imread("images/mask15.tif")

#Spliting the image in two
yMax = img_src.shape[1] - img_src.shape[1]%Int_Win # Clipping the end
xMax = int(img_src.shape[0]/2 - img_src.shape[0]/2%Int_Win)
img_a = img_src[:xMax, :yMax]
img_b = img_src[1236:(1236+xMax), :yMax]

# Slice the images into interrogation windows
def slice(image):
    sliced = np.empty((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))), dtype=object)
    xc = 0
    for x in range(0,xMax,int((1-(Overlap/100))*Int_Win)):
        yc = 0
        for y in range(0, yMax, int((1-(Overlap/100))*Int_Win)):
            sliced[xc][yc] = np.array(image[x:(x+Int_Win), y:(y+Int_Win)])
            yc+=1
        xc+=1
    return sliced

slice_a = slice(img_a)
slice_b = slice(img_b)
if AoA == 0:
    slice_m0 = slice(mask0)
elif AoA == 5:
    slice_m0 = slice(mask5)
elif AoA == 15:
    slice_m0 = slice(mask15)
else:
    print("Angle of attack input error")

# Arrays with x and y velocities
dys = np.zeros((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))))
dxs = np.zeros((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))))
dyst = np.zeros((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))))
dxst = np.zeros((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))))

qual = 3

# The cross correlation function
def cross_corr(sl_img_a, sl_img_b, x, y):

    corr = signal.correlate(sl_img_a[x][y] - sl_img_a[x][y].mean(), sl_img_b[x][y] - sl_img_b[x][y].mean(), method="fft")/(np.std(sl_img_a[x][y])*np.std(sl_img_b[x][y]))
    smooth = interpolate.interp2d(np.linspace(0,1, 63),np.linspace(0,1, 63), corr, kind="cubic")
    y,x = np.unravel_index(np.asarray(smooth(np.linspace(0,1, qual*63),np.linspace(0,1, qual*63))).argmax(), (qual*63,qual*63))
    dy, dx = y/qual-(Int_Win - 1), x/qual-(Int_Win - 1)

    sec = np.partition(corr.flatten(), -2)[-2]
    SNR = corr.max()/sec # Signal to noise ratio
    return dy, dx

#V1
# Run the cross correlation for all interrogation windows
if Overlap==0:
    for x in range(slice_a.shape[0]-1):
        for y in range(slice_a.shape[1]-1):
            dy, dx = cross_corr(slice_a, slice_b, x,y) #have to take the average
            dys[x][y] = dy
            dxs[x][y] = dx
else:
    # Run the cross correlation for all interrogation windows
    for x in range(slice_a.shape[0]-1):
        for y in range(slice_a.shape[1]-1):
            dy, dx = cross_corr(slice_a, slice_b, x,y) #have to take the average
            dyst[x][y] = dy
            dxst[x][y] = dx
            if ((x==0 and y==0) or (x==0 and y==(slice_a.shape[1]-1)) or (x==(slice_a.shape[0]-1) and y==0) or (x==(slice_a.shape[0]-1) and y==(slice_a.shape[1]-1))):
                dys[x][y] = dy
                dxs[x][y] = dx
            elif (y==0 or y==(slice_a.shape[1]-1)):
                dys[x][y] = (dy + dyst[x-1][y])/2
                dxs[x][y] = (dx + dxst[x-1][y])/2
            elif (x==0 or x==(slice_a.shape[0]-1)):
                dys[x][y] = (dy + dyst[x][y-1])/2
                dxs[x][y] = (dx + dxst[x][y-1])/2
            else:
                dys[x][y] = (dy + dyst[x][y-1] + dyst[x-1][y] + dyst[x-1][y-1])/4
                dxs[x][y] = (dx + dxst[x][y-1] + dxst[x-1][y] + dxst[x-1][y-1])/4

dxs = dxs*-1

# Apply mask
if mask_on == 1:
    for i in range(slice_m0.shape[0]-3):
        for j in range(slice_m0.shape[1]-3):
            count = 0
            for k in range(slice_m0[i][j].shape[0]):
                for l in range(slice_m0[i][j].shape[1]):
                    if max(slice_m0[i][j][k][l]) == 255:
                        count += 1
                        break
            if count>=1:
                dys[i][j] = 0
                dxs[i][j] = 0
elif mask_on != 0:
    print("Mask on input error")

vect = np.sqrt(dxs**2 + dys**2)*pix_vel

cmap = plt.get_cmap('jet')
cmap.set_under("black")  # Color for values less than vmin

# Plot the vector field
fig, ax = plt.subplots()
ax.quiver(dxs[:-2, :-2],dys[:-2, :-2],vect[:-2, :-2],scale_units="xy",scale=5, cmap=cmap)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.invert_xaxis()
plt.show()

# Plot the velocity magnitude
plt.imshow(np.flip(vect[:-2, :-2], 0), vmin=0.001, vmax=14, cmap=cmap)
plt.colorbar()
plt.show()


# Showing images
# plt.imshow(img_src_cal, cmap="gray", vmin=0, vmax=10000)
# plt.show()
# plt.imshow(img_a, vmin=0, vmax = 300, cmap="gray")
# plt.show()
# plt.imshow(img_overlay, vmin=0, vmax = 300, cmap="gray")
# plt.show()

# Plot the cross correlation matrix
# x = np.arange(-31,32,1)
# y = np.arange(-31,32,1)
# X,Y = np.meshgrid(x,y)
#
#
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
#
#
# # Plot a 3D surface
# ax.plot_surface(X, Y, cc)
#plt.show()
# overlay = cv.addWeighted(img_a, 0.5, img_b, 0.5, 0)
# plt.imshow(overlay, cmap="gray", vmin=0, vmax=300)
# plt.show()
# overlay = cv.addWeighted(img_test[:xMax, :yMax], 0.5, img_test[1236:(1236+xMax), :yMax], 0.5, 0)
# plt.imshow(overlay, cmap="gray", vmin=0, vmax=300)
# plt.show()
