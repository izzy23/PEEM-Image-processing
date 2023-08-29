import numpy as np
import cv2
import math
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave
from matplotlib.widgets import RadioButtons
import scipy.optimize as scipy
from skimage import feature
from skimage.feature import canny
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


imageNumbers = np.arange(0, 7)

differenceImages = []
intensityImages = []

for i in imageNumbers: 

    differenceImg = io.imread(r"finalCross/alignedCrossNewCentreTest22_D_%s.tif" % str(i))
    intensityImg = io.imread(r"finalCross/alignedCrossNewCentreTest22_I_%s.tif" % str(i))

    differenceImg = np.array(differenceImg)
    checkImg = np.clip(differenceImg, -0.08, 0.1)
    differenceImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    differenceImg = differenceImg * 255

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

differenceImages = np.array(differenceImages, dtype = "uint8")


for i in range(0, len(differenceImages)):
    plt.imshow(differenceImages[i], cmap="gray")
    plt.show()

    #estimates standard deviation of image pixel values
    sigma_est = estimate_sigma(differenceImages[i], average_sigmas=True)

    #finds edges in image
    edgeImg = cv2.Canny(differenceImages[i], 32, 39, None, 3,)
    #edgeImg = feature.canny(differenceImages[i], sigma = 1)

    plt.imshow(edgeImg, cmap="gray")
    plt.show()
    lines = cv2.HoughLines(edgeImg, 1, np.pi/180, 150, None, 0, 0)
     # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(differenceImages[i], pt1, pt2, (0,0,255), 3, cv2.LINE_AA)