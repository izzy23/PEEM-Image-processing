from skimage import io
from skimage.io import imsave as sk_imsave

import numpy as np
import h5py
import cv2
import time
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
import skimage
from skimage import io
imageNumbers = np.arange(1, 6)
differenceImages = [io.imread(r"finalCross/alignedCrossNewFinal2_D_0.tif")]
intensityImages = [io.imread(r"finalCross/alignedCrossNewFinal2_I_0.tif")]



for i in imageNumbers: 

    differenceImg = io.imread(r"finalCross/alignedCrossNewFinal2_D_%s.tif" %str(i))
    intensityImg = io.imread(r"finalCross/alignedCrossNewFinal2_I_%s.tif" %str(i))
    print("shape")
    print(differenceImg.shape)
    print(intensityImg.shape)
    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

displayImg = np.clip(differenceImages[0], -0.08, 0.1)
displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

# Select ROI
w, h = displayImg.shape

# Naming a window
cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

#resizes window, not actual image
cv2.resizeWindow("select ROI", w , h)

r = cv2.selectROI("select ROI", displayImg)

checkInitialImg = differenceImages[0][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
initialAvg = np.nanmean(checkInitialImg)
initialStd = np.nanstd(checkInitialImg)

for i in range(1, len(differenceImages)):
    image = differenceImages[i][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    avg = np.nanmean(image)
    std = np.nanstd(image)

    image = differenceImages[i]

    newImage = initialAvg + ((image - avg) * (initialStd / std))
    differenceImages[i] = newImage


plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)


img1 = differenceImages[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)


#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")

print("plotted")
sb.on_changed(update)

plt.show()

for i in range(0, len(differenceImages)):
    sk_imsave("finalCross/alignedCrossNewFinal4_D_%s.tif" %str(i), differenceImages[i])
    sk_imsave("finalCross/alignedCrossNewFinal4_I_%s.tif" %str(i), intensityImages[i])