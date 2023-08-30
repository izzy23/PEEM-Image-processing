
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
    
# open the file as 'f'
with h5py.File('Diamond images\medipix-321130.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]
    #print("key = " + str(f.keys()))
    #data = np.array(f[key]["data"]["data"])        #inital definition gave data = ["data" "instrument"]
    data = np.array(f[key]["instrument"]["detector"]["data"])

    #img = data

    img = data[0][0]
    v = np.arange(0, 512)

    plt.axis("off")

    ax1 = plt.subplot(2, 2, 1)
    #fig = plt.imshow(img, cmap="gray", vmin=300, vmax = 700)
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

    #plt.axis("off")
    sb.on_changed(update)

    #plt.show()


    def averageImages(data, energyIndex):
        imStack = data[energyIndex][:]
        imageSum = imStack[0]
        for image in range(1, len(imStack)):
            imageSum = imageSum + imStack[image]
        imageSum = imageSum / len(imStack)

        return imageSum

        #fig2 = plt.imshow(imageSum, cmap="gray", vmin=300, vmax = 700)
        #ax2 = plt.subplot(2, 2, 2)
        #ax2.imshow(imageSum, cmap="gray", vmin=300, vmax = 700)
    
    def imgDiff(data):
        avgE1 = averageImages(data, 0)
        avgE2 = averageImages(data, 1)

        difference = (avgE1 - avgE2) / (avgE1 + avgE2)
        return difference

    ax1.set_title("image stack E1")
    #for average at 1st energy
    imageSumE1 = averageImages(data, 0)

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(imageSumE1, cmap="gray", vmin=300, vmax = 700)
    ax2.set_title("Average E1")

    #for difference
    imageAvgDiff = imgDiff(data)
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(imageAvgDiff, cmap="gray", vmin=-0.15, vmax = 0.15)
    ax3.set_title("difference")


    #for average at 2nd energy
    imageSumE2 = averageImages(data, 1)
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(imageSumE2, cmap="gray", vmin=300, vmax = 700)
    ax4.set_title("Average E2")
    plt.tight_layout()
    plt.show()
        

    #time.sleep(8)
    #f.close()
