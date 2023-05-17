import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Loading images
img_src = cv.imread("images/alpha15_Diff/B00001.tif")
img_src_cal = cv.imread("images/Calibration_02/B00001.tif")[1236:, :]

#Spliting the image in two
img_a = img_src[:1236, :]
img_b = img_src[1236:, :]

#Correcting exposure and brightnes and combining both images
img_cal = cv.addWeighted(img_src_cal, 2.5, img_src_cal, 2.5, 4)
img = cv.addWeighted(img_a, 100, img_b, 100, 30)

#Showing images
plt.imshow(img_cal)
plt.show()

plt.imshow(img)
plt.show()
