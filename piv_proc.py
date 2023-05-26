import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import ndimage
import math

# Defining the variables
Int_Win = 32               # Interrogation window size in pixles
Overlap = 50               # Overlap of interrogation windows in %
dt = 72*10**-6

# Calibration
p1 = [300.8, 317.0]
p2 = [1236.3, 335.9]
pix_dist = math.dist(p1, p2)
pix_vel = 0.11/(pix_dist*dt)

#Loading images
img_src = cv.imread("images/B00010.tif", -1)
img_src_cal = cv.imread("images/Calibration_02/B00001.tif", -1)[1236:, :]

#Spliting the image in two
yMax = img_src.shape[1] - img_src.shape[1]%Int_Win # Clipping the end
xMax = int(img_src.shape[0]/2 - img_src.shape[0]/2%Int_Win)
img_a = img_src[:xMax, :yMax]
img_b = img_src[1236:(1236+xMax), :yMax]

#Creating an overlay
img_overlay = cv.addWeighted(img_a, 0.5, img_b, 0.5, 0)

ys = np.arange(int(Int_Win/2), yMax-int(Int_Win/2)+1, int((1-Overlap/100)*Int_Win))
xs = np.arange(int(Int_Win/2), xMax-int(Int_Win/2)+1, int((1-Overlap/100)*Int_Win))

# Slice the images into interrogation windows
def slice(image):
    sliced = np.empty((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))), dtype=object)

    xc = 0
    for x in range(0,xMax,int((1-(Overlap/100))*Int_Win)):
        yc = 0
        for y in range(0, yMax, int((1-(Overlap/100))*Int_Win)):

            sliced[xc][yc] = np.array(image[x:(x+Int_Win), y:(y+Int_Win)])

            #print(x, (x+Int_Win), y, (y+Int_Win))
            yc+=1
        xc+=1
    return sliced

slice_a = slice(img_a)
slice_b = slice(img_b)

dys = np.zeros((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))))
dxs = np.zeros((int(xMax/((1-(Overlap/100))*Int_Win)), int(yMax/((1-(Overlap/100))*Int_Win))))


# The cross correlation function
def cross_corr(sl_img_a, sl_img_b, x, y):
    corr = signal.correlate(sl_img_a[x][y] - sl_img_a[x][y].mean(), sl_img_b[x][y] - sl_img_b[x][y].mean(), method="fft")/(np.std(sl_img_a[x][y])*np.std(sl_img_b[x][y]))
    y,x = np.unravel_index(corr.argmax(), corr.shape)
    dy, dx = y-(Int_Win - 1), x-(Int_Win - 1)
    sec = np.partition(corr.flatten(), -2)[-2]
    SNR = corr.max()/sec
    return dy, dx

# Run the cross correlation for all interrogation windows
for x in range(slice_a.shape[0]):
    for y in range(slice_a.shape[1]):
        dy, dx = cross_corr(slice_a, slice_b, x,y)
        dys[x][y] = dy
        dxs[x][y] = -dx

vect = np.sqrt(dxs**2 + dys**2)*pix_vel
for i in range(len(vect)):
    for j in range(len(vect[i])):
        if abs(vect[i][j])>12:
            vect[i][j] = 0

# Plot the vector field
fig, ax = plt.subplots()
ax.quiver(dxs,dys,vect,cmap="plasma",angles="xy",scale_units="xy",scale=4,)
ax.set_aspect("equal")
plt.show()

# Plot the velocity magnitude
plt.imshow(vect[:74, :98], vmin=0, vmax=12, cmap="jet")
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
