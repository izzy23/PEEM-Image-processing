from skimage import io
from skimage.io import imsave as sk_imsave
import numpy as np
import time
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
import skimage
from skimage import io
imageNumbers = np.arange(1, 7)
differenceImages = [io.imread(r"AlignedCross\alignedCross_D_0.tif")]
intensityImages = [io.imread(r"AlignedCross\alignedCross_I_0.tif")]

for i in imageNumbers: 

    differenceImg = io.imread(r"AlignedCross\alignedCross_D_%s.tif" %str(i))
    intensityImg = io.imread(r"AlignedCross\alignedCross_I_%s.tif" %str(i))
    
    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)

img1 = differenceImages[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)


#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")


sb.on_changed(update)

plt.show()
