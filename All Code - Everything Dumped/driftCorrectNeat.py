import numpy as np
import h5py
import cv2
#import time
#import io
#import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d



def interpolateImages(imgArray):
    interpolatedImages = []

    for currentImg in imgArr:
        
        #img = cv2.imread(currentImg)
        #normalise - so all pixel values less than 1
        currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        
        img = currentImg

        #changed fx and fy = 10 to 2 so maybe quicker for now
        #This gives sub-pixel resolution
        #inter linear is just linear, can switch - LINEAR RECOMENDED BY WEBSITE
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_LINEAR)
        interpolatedImage = cv2.resize(img, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_LANCZOS4)
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 6, fy = 6, interpolation = cv2.INTER_CUBIC)

        #adjusting contrast - so differences can be picked up, and aren't lost when converted to 8-bit later
        #these values worked
        #displayMin = 0.001
        #displayMax = 0.1

        displayMin = 0.043
        displayMax = 0.098
        interpolatedImage.clip(displayMin, displayMax, out=interpolatedImage) #hopefully adjust contraast

        interpolatedImage = (interpolatedImage-np.min(interpolatedImage))/(np.max(interpolatedImage)-np.min(interpolatedImage)) #renormalises - after contrast adjust so shows
        
        interpolatedImages = interpolatedImages + [interpolatedImage]
    
    return interpolatedImages

def choseROI(firstImg):
    # Select Reigon Of Interest

    w, h = firstImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)
    
    r = cv2.selectROI("select ROI", firstImg)
    
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage


def templateMatch(img, croppedImg):

    

    img = img * 255
    img = np.array(img, dtype="uint8")
    #img = cv2.Canny(img,200,400)
    #img = cv2.Canny(img, 100, 120)

    #plt.imshow(img)
    #plt.show()
    croppedImg = croppedImg * 255
    croppedImg = np.array(croppedImg, dtype="uint8")
    #croppedImg = cv2.Sobel(img,100,120)

    #applys OpenCV's template matching function
    #finds convolution between ROI image and other full image at all positions
    #should use CCOEFF not CCORR as looks like intensities change between images
    #matchMatrix = cv2.matchTemplate(img, croppedImg, cv2.TM_CCOEFF_NORMED)
    matchMatrix = cv2.matchTemplate(img, croppedImg, cv2.TM_CCOEFF)

    #plot convolution
    
    #x = np.arange(0, matchMatrix.shape[1])
    #y = np.arange(0, matchMatrix.shape[0])
    #xg, yg = np.meshgrid(x, y)
    #figNew = plt.figure()
    #axNew = plt.axes(projection="3d")
    #axNew.plot_surface(xg, yg, matchMatrix, cmap="viridis")
    #plt.show()

    #finds positions with max similarity score between cropped image and second image 
    loc = np.where(matchMatrix == np.max((matchMatrix)))
    
    #x and y positions in new image where best match
    return(loc[1], loc[0])

def templateMatchStack(imStack, croppedImg):
    positions = []

    #testing given offset - 
    imgW, imgH = imStack[3].shape

    imStack[3] = translateImage(imgH, imgW, imStack[3], [0, 0], [10, 10])
    #plt.imshow(imStack[3], cmap="gray")
    #plt.show()
    
    for image in imStack:

        pos = templateMatch(image, croppedImg)
        positions = positions + [[pos[0][0], pos[1][0]]]

    print("outputting positions")
    print(positions)

    
    return positions

def translateImage(imgH, imgW, img, fixedPos, driftPos):

    #difference in x and y positions for ROI between images
    tx = driftPos[0] - fixedPos[0]
    ty = driftPos[1] - fixedPos[1]

    print("image shifts")
    print("x shift = " + str(tx))
    print("y shift = " + str(ty))

    #generates translation matrix
    translationMatrix = np.array([
    [1, 0, -tx],
    [0, 1, -ty]
    ], dtype=np.float32)

    #apply transformation
    translatedImage = cv2.warpAffine(src=img, M=translationMatrix, dsize=(imgW, imgH))
    return translatedImage

def translateStack(imgH, imgW, imageStack, positions):
    fixedPos = positions[0]
    correctedImages = [imageStack[0]]

    for i in range(1, len(imageStack)):
        currentImage = imageStack[i]
        correctedImages = correctedImages + [translateImage(imgH, imgW, currentImage, fixedPos, positions[i])]

    return correctedImages

def averageImages(imStack):

        imageSum = imStack[0]

        #adding pixel vals for each image
        for image in range(1, len(imStack)):
            imageSum = imageSum + imStack[image]

        #becomes average
        imageSum = imageSum / len(imStack)

        return imageSum

def driftCorrect(imgArr):
    imageStack = interpolateImages(imgArr)

    croppedImage = choseROI(imageStack[0])  #select ROI from 1st image in stack

    #Display cropped reigon of interest chosen
    #plt.imshow(croppedImage,cmap="gray")
    #plt.title("Selected ROI")
    #plt.show()

    #finds positions of selected reigon of interest in all images in stack
    positions = templateMatchStack(imageStack, croppedImage)

    imgH, imgW = imageStack[1].shape[:2]

    correctedStack = translateStack(imgH, imgW, imageStack, positions)
    print("length of corrected stack")
    print(len(correctedStack))
    return correctedStack

# open the file as 'f'
#with h5py.File('Diamond images\medipix-321127.hdf', 'r') as f:
with h5py.File('Diamond images\medipix-321130.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]
    data = np.array(f[key]["instrument"]["detector"]["data"])

    #array of images from file
    imgArr = data[0][:]

    correctedStack = driftCorrect(imgArr)

    plt.axis("off")

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    img1 = imgArr[0]
    img2 = correctedStack[0]

    fig1 = ax1.imshow(img1, cmap="gray", vmin = 300, vmax=700)
    fig2 = ax2.imshow(img2, cmap="gray",  vmin = 0, vmax=1)


    #defines slider axis
    axs = plt.axes([0.15, 0.001, 0.65, 0.03])
    sb = Slider(axs, 'image no', 0, 20, valinit=0, valstep = np.arange(0, 20))

    #runs when slider moved
    def update(val):
        img1 = imgArr[sb.val]
        img2 = correctedStack[sb.val]

        fig1.set_data(img1)
        fig2.set_data(img2)
        
        plt.draw()

    sb.on_changed(update)
    plt.show()
    


    

    