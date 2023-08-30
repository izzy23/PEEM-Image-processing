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

differenceImg = io.imread(r"C254 - Cross Centre/321627-321636_Diffference.tif").astype(np.float32)
intensityImg = io.imread(r"C254 - Cross Centre/321627-321636_Intensity.tif").astype(np.float32)


plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

ax1.imshow(differenceImg, cmap="gray", vmin = -0.08, vmax = 0.1) #both were 0.1
ax2.imshow(intensityImg, cmap="gray", vmin = 2, vmax = 30)  #30 was 25

plt.show()