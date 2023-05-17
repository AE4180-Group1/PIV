import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img_src = cv.imread("images/alpha15_Diff/B00001.tif")
img_src_cal = cv.imread("images/Calibration_02/B00001.tif")


img_a = img_src[:1236, :]
img_b = img_src[1236:, :]
img_cal = cv.addWeighted(img_src_cal, 0, img_src_cal, 5, 4)
img = cv.addWeighted(img_a, 100, img_b, 100, 30)

plt.imshow(img_cal)
plt.show()


plt.imshow(img)
plt.show()
