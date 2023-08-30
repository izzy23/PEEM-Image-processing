
import numpy as np
import h5py
#import peempy
from peempy.imageproc import DriftCorrector
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
    
# open the file as 'f'
with h5py.File('Diamond images\medipix-321127.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]

    data = np.array(f[key]["instrument"]["detector"]["data"])

    img = data[0][0]
    v = np.arange(0, 512)

    plt.axis("off")

    ax1 = plt.subplot(2, 2, 1)

    fig1 = ax1.imshow(img, cmap="gray", vmin=300, vmax = 700)

    #defines slider axis
    #axs = plt.axes([0.15, 0.1, 0.65, 0.03])
    axs = plt.axes([0.15, 0.001, 0.65, 0.03])
    sb = Slider(axs, 'image no', 0, 20, valinit=0, valstep = np.arange(0, 20))

    #runs when slider moved
    def update(val):
        img = data[0][sb.val]

        #fig.set_data(img)
        fig1.set_data(img)
        
        #fig1.draw()
        plt.draw()

    sb.on_changed(update)

    #plt.show()
    def averageImages(data, energyIndex):

        #testing non drift corrected stack here
        imStack = data[energyIndex][:]

        imageSum = imStack[0]

        #adding pixel vals for each image
        for image in range(1, len(imStack)):
            imageSum = imageSum + imStack[image]

        #becomes average
        imageSum = imageSum / len(imStack)

        return imageSum

    def averageImagesDriftCorrected(data, energyIndex):

        #testing drift corrected stack here
        imStack = data[energyIndex][:]
        #imStack = peempy.imageproc.DriftCorrector(imStack) #I think i may have fixed the libary, but don't try displaying things - removed that part
        imstack = DriftCorrector(imStack)

        imageSum = imStack[0]

        #adding pixel vals for each image
        for image in range(1, len(imStack)):
            imageSum = imageSum + imStack[image]

        #becomes average
        imageSum = imageSum / len(imStack)

        return imageSum
    
    def imgDiff(data):

        avgE1 = averageImages(data, 0)
        avgE2 = averageImages(data, 1)

        difference = (avgE1 - avgE2) / (avgE1 + avgE2)
        return difference

    ax1.set_title("image stack E1")
    #for average at 1st energy

    imageSumE1 = averageImages(data, 0)

    ax2 = plt.subplot(2, 1, 1)
    ax2.imshow(imageSumE1, cmap="gray", vmin=300, vmax = 700)
    ax2.set_title("Average E1 not drift corrected")


    avgDriftCorrect = averageImagesDriftCorrected(data, 0)

    ax3 = plt.subplot(2, 1, 2)
    ax3.imshow(avgDriftCorrect, cmap="gray", vmin=300, vmax = 700)
    ax3.set_title("Average E1 drift corrected")
    
    plt.tight_layout()
    plt.show()

    