import numpy as np
import h5py
import cv2
#import time
import io
#import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave
import random

def interpolateImages(imgArray):

    interpolatedImages = []

    for currentImg in imgArray:
        
        #img = cv2.imread(currentImg)
        #normalise - so all pixel values less than 1
        currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        
        img = currentImg
        interpolatedImage = cv2.resize(img, (0, 0), fx = 1, fy = 1, interpolation = cv2.INTER_LANCZOS4)

        interpolatedImage = (interpolatedImage-np.min(interpolatedImage))/(np.max(interpolatedImage)-np.min(interpolatedImage)) #renormalises - after contrast adjust so shows
        
        interpolatedImages = interpolatedImages + [[interpolatedImage.astype(np.float32)]]

    return interpolatedImages

def choseROI(firstImg):
    # Select ROI
    w, h = firstImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    firstImg = firstImg * 255


    firstImg = np.array(firstImg, dtype = "uint8")

    checkImg = np.clip(firstImg, 100, 150)
    checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    r = cv2.selectROI("select ROI", checkImg)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):
    img = img * 255
    img = np.array(img, dtype="uint8")

    #applys OpenCV's template matching function
    #finds convolution between ROI image and other full image at all positions
    matchMatrix = cv2.matchTemplate(img, croppedImg, cv2.TM_CCOEFF_NORMED)

    #finds positions with max similarity score between cropped image and second image 
    loc = np.where(matchMatrix == np.max(matchMatrix))

    print("match similarity")
    print(np.nanmax(matchMatrix))
    
    #x and y positions in new image where best match
    return(loc[1], loc[0])

def templateMatchStack(imStack, croppedImg):
    positions = []
    for image in imStack:
        pos = templateMatch(image, croppedImg)
        positions = positions + [[pos[0][0], pos[1][0]]]

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

def standardizeStack(imgStack):
    arr = []
    standardDeviations = []

    for img in imgStack:
         arr = arr + imgStack
         currentStandardDeviation = np.std(arr)
         standardDeviations = standardDeviations + [currentStandardDeviation]
    standardDeviation = np.average(standardDeviations)
    arr = np.array(arr, dtype = np.float32)
    mean = arr / len(imgStack)

    newStack = []
    for img in imgStack:
         img  = (img - mean) / standardDeviation
         newStack = newStack + [[img]]

    
    return newStack

def driftCorrect(imgArr):


    imageStack = np.array(imgArr, dtype=np.float32)
    
    croppedImage = choseROI(imageStack[0])  #select ROI from 1st image in stack


    #finds positions of selected reigon of interest in all images in stack
    positions = templateMatchStack(imageStack, croppedImage)

    #imgH, imgW = imageStack[1].shape[:2]
    imgH, imgW = imageStack[1].shape
    imgH = len(imageStack[1][0][:])
    imgW = len(imageStack[1][:][0])
    print("image height and width")
    print(imgH)
    print(imgW)


    correctedStack = translateStack(imgH, imgW, imageStack, positions)

    return correctedStack

def averageImages(imStack):

    imageSum = imStack[0]

    #adding pixel vals for each image
    for image in range(1, len(imStack)):
        
        #can't remove float - 8bit wrap around weird thing
        imageSum = imageSum.astype(float) + imStack[image].astype(float)
        #imageSum = imageSum + imStack[image]

    #becomes average
    imageSum = imageSum / len(imStack)

    return imageSum


startNo = 0
endNo = 6

imageNumbers = np.arange(startNo + 1, endNo + 1)

differenceImages = [io.imread(r"rotatedCross\%s_D.tif" % str(startNo)).astype(np.float32)]
intensityImages = [io.imread(r"rotatedCross\%s_I.tif" % str(startNo)).astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"rotatedCross\%s_D.tif" % str(i))
    intensityImg = io.imread(r"rotatedCross\%s_I.tif" % str(i))

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]


initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    print("differenceMin = " + str(np.nanmin(image)))
    print("differenceMax = " + str(np.nanmax(image)))
    avg = np.nanmean(image)
    std = np.nanstd(image)
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    image = cv2.medianBlur(image, 5)

    imageStack = imageStack + [image]

differenceImages = imageStack
print("no of images")
print(len(differenceImages))

initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []


for image in intensityImages:
    print("looping")
    avg = np.nanmean(image)
    std = np.nanstd(image)

    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))


    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack
print("length before corrections")
print(len(intensityImages))


differenceImages = np.array(differenceImages, dtype=np.float32)
intensityImages = np.array(intensityImages, dtype=np.float32)
    
croppedImage = choseROI(differenceImages[0])  #select ROI from 1st image in stack


#finds positions of selected reigon of interest in all images in stack
#positionsD = templateMatchStack(differenceImages, croppedImage)

zoomMin = 0.5
zoomMax = 1.5
centreVal = (zoomMax + zoomMin) / 2
complete = False

while complete == False:
    
    matchMatrix = cv2.matchTemplate(img, croppedImg, cv2.TM_CCOEFF_NORMED)

    #finds positions with max similarity score between cropped image and second image 
    loc = np.where(matchMatrix == np.max(matchMatrix))



#imgH, imgW = imageStack[1].shape[:2]
#imgH, imgW = differenceImages[1].shape
imgH = len(differenceImages[1][0][:])
imgW = len(differenceImages[1][:][0])



correctedDifferences = translateStack(imgH, imgW, initialDifferenceImages, positionsD)

correctedIntensities = translateStack(imgH, imgW, initialIntensityImages, positionsD)
print("length after correction")
print(len(correctedIntensities))

#correctedDifferences = driftCorrect(differenceImages)
#correctedIntensities = driftCorrect(intensityImages)
plt.figure()
plt.axis("off")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

img1 = correctedDifferences[0]
img2 = correctedIntensities[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.1, vmax = 0.1)
fig2 = ax2.imshow(img2, cmap="gray", vmin = 2, vmax = 25)


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferences[sb.val]
    img2 = correctedIntensities[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("differences")
ax2.set_title("intensities")

averageIntensity = averageImages(correctedIntensities)
averageDifference = averageImages(correctedDifferences)

ax4.imshow(averageIntensity, cmap="gray", vmin = 2, vmax = 25)
ax3.imshow(averageDifference, cmap="gray", vmin = -0.1, vmax = 0.1)
print("plotted")
sb.on_changed(update)
plt.show()
