import numpy as np
import h5py
import cv2
#import time
#import io
#import os
from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib.cm as cm  
#from matplotlib.widgets import Slider

def interpolateImages(imgArray):
    interpolatedImages = []

    for currentImg in imgArr:
        
        #img = cv2.imread(currentImg)
        #normalise - so all pixel values less than 1
        currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        
        #currentImg = currentImg * 225  #DON'T DO THI - it breaks everything
        #img = Image.fromarray(np.uint8(cm.gist_earth(currentImg)*255))
        img = currentImg
        #cv2.imshow("currentImg", currentImg)   #this works now - did actually display grey image this morning, but stuck on white screen now - maybe slow?

        #changed fx and fy = 10 to 2 so maybe quicker for now
        interpolatedImage = cv2.resize(img, (0, 0), fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
        #print("before clipping")
        #plt.imshow(interpolatedImage)
        #plt.show()


        #adjusting contrast - so differences can be picked up
        displayMin = 0.001
        displayMax = 0.1
        interpolatedImage.clip(displayMin, displayMax, out=interpolatedImage) #hopefully adjust contraast
        #plt.imshow(interpolatedImage)
        #plt.show()
        interpolatedImage = (interpolatedImage-np.min(interpolatedImage))/(np.max(interpolatedImage)-np.min(interpolatedImage)) #renormalises - after contrast adjust so shows
        #plt.imshow(interpolatedImage, cmap="gray")
        #print("interpolated")
        #plt.show()

        #to adjust contrast
        #newImage = []
        
        interpolatedImages = interpolatedImages + [interpolatedImage]
        print("shape of interpolated images")
        
        print(len(interpolatedImages))
    print(len(interpolatedImages[0]))
    
    return interpolatedImages

def choseROI(firstImg):
    # Select ROI
    #firstImg = Image.fromarray(firstImg, "L")
    #firstImg = np.array(firstImg, dtype="uint8")
    #contrast = 1
    #brightness= 0.2
    #firstImg = cv2.addWeighted( firstImg, contrast, firstImg, 0, brightness)

    #firstImg = cv2.convertScaleAbs(firstImg, alpha=0, beta=150)
    r = cv2.selectROI("select ROI", firstImg)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #plt.imshow(croppedImage)
    #plt.show()

    return croppedImage

def templateMatch(img, croppedImg):

    #converts so 8-bit array - required to convert to greyscale
    #print("no of chanels -  full img")
    #print(img[0][0].shape)
    #print("no of channels - cropped img")
    #print(croppedImg[0][0].shape)

    #if img.shape == 3:
    #    img = np.array(img, np.uint8)
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if croppedImg.shape == 3:
    #    croppedImg = np.array(croppedImg, np.uint8)
    #    croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
    #plt.title("before anythin")
    #plt.imshow(img)
    #plt.show()
    #print("image from array")
    #img = Image.fromarray(img, "L")
    #plt.imshow(img)
    #plt.title("image from array")
    #plt.show()

    #print("max and min values before datatype conversion")
    #print(np.max(img))
    #print(np.min(img))
    #finds contours on image to match to 

    #plt.title("datatype conversion")

    #img = np.array(img, dtype="uint8")
    img = img * 255
    img = np.array(img, dtype="uint8")
    #plt.imshow(img, cmap="gray")
    #plt.show()
    #croppedImg = Image.fromarray(croppedImg, "L")
    croppedImg = croppedImg * 255
    croppedImg = np.array(croppedImg, dtype="uint8")

    #attempt at contrast adjust - doesn't work
    #contrast = 2
    #brightness= 5
    #img = cv2.addWeighted( img, contrast, img, 0, brightness)
    #croppedImg = cv2.addWeighted( croppedImg, contrast, croppedImg, 0, brightness)

    #converts image to greyscale - needed for template match

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)

    #applys OpenCV's template matching function
    #finds convolution between ROI image and other full image at all positions
    matchMatrix = cv2.matchTemplate(img, croppedImg, cv2.TM_CCOEFF_NORMED)

    #finds positions with max similarity score between cropped image and second image 
    loc = np.where(matchMatrix == np.max(matchMatrix))
    
    #x and y positions in new image where best match
    return(loc[1], loc[0])

def templateMatchStack(imStack, croppedImg):
    positions = []
    for image in imStack:
        pos = templateMatch(image, croppedImg)
        positions = positions + [[pos[0][0], pos[1][0]]]
    print("outputting positions")
    print(positions)
    return positions


# open the file as 'f'
with h5py.File('Diamond images\medipix-321127.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]
    data = np.array(f[key]["instrument"]["detector"]["data"])

    imgArr = data[0][:]
    interpolatedImages = []

    imageStack = interpolateImages(imgArr)
    #cv2.imshow("image stack", imageStack[0])

    croppedImage = choseROI(imageStack[0])  #select ROI from 1st image in stack

    #Display cropped reigon of interest chosen
    plt.imshow(croppedImage)
    plt.imshow(imageStack[0])
    plt.show()
    positions = templateMatchStack(imageStack, croppedImage)
    #print(templateMatch(np.array(imageStack[0]), np.array(croppedImage))) #maybe convert to 8 bit here
    
    plt.imshow(imageStack[0],cmap="gray")
    plt.show()