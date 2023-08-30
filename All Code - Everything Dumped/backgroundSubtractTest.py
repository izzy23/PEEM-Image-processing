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




def translateImage(imgH, imgW, img, fixedPos, driftPos):

    #difference in x and y positions for ROI between images
    tx = driftPos[0] - fixedPos[0]
    ty = driftPos[1] - fixedPos[1]

    #print("image shifts")
    #print("x shift = " + str(tx))
    #print("y shift = " + str(ty))

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


imageNumbers = np.arange(321128, 321137)

differenceImages = [io.imread(r"writeImages\321127_D_avg.tif").astype(np.float32)]
intensityImages = [io.imread(r"writeImages\321127_I_avg.tif").astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"writeImages\%s_D_avg.tif" % str(i))
    intensityImg = io.imread(r"writeImages\%s_I_avg.tif" % str(i))
    #print("shape")
    #print(differenceImg.shape)
    #print(intensityImg.shape)

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

imageStack = []
initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    #print("differenceMin = " + str(np.nanmin(image)))
    #print("differenceMax = " + str(np.nanmax(image)))
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #newImage = avg + ((image - initialAvg) * (std / initialStd))
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    image = cv2.medianBlur(image,3)

    imageStack = imageStack + [image]

differenceImages = imageStack
print("num diff images")
print(len(differenceImages))
initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []

for image in intensityImages:
    print("adding intensity images")
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #print("initial mean " + str(avg))
    #print("initial standard deviation " + str(std))
    image = np.clip(image, 0, 30)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))


    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    
    image = cv2.medianBlur(image,3)

    #image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack
print("len intenisty images")
print(len(intensityImages))


#differenceImages = np.array(differenceImages, dtype=np.float32)
#intensityImages = np.array(intensityImages, dtype=np.float32)

imgH = len(differenceImages[1][0][:])
imgW = len(differenceImages[1][:][0])

checkRange = 10
currentMean = 999999999

initialDiffImg = differenceImages[2][:][:]
correctedDifferences = []

for image in differenceImages:
    print("new img")
    for xMovement in range(-checkRange, checkRange):
        for yMovement in range(-checkRange, checkRange):
            checkImg = translateImage(imgH, imgW, image, [0, 0], [xMovement, yMovement] )
            subtractedImg = checkImg - initialDiffImg
            if abs(np.mean(subtractedImg)) < currentMean:
                currentMean = abs(np.mean(subtractedImg))
                xTrans = xMovement
                yTrans = yMovement
    correctedImage = translateImage(imgH, imgW, image, [0, 0], [xTrans, yTrans])
    correctedDifferences = correctedDifferences + [correctedImage]
print("no output images = " + str(len(correctedDifferences)))

#correctedDifferences = translateStack(imgH, imgW, differenceImages, [0, 0], [xTrans, yTrans])
print("differeces done")
currentMean = 999999999
correctedIntensities = []
initialIntensityImg = intensityImages[0]
for image in intensityImages:
    print("new img")
    for xMovement in range(-checkRange, checkRange):
        for yMovement in range(-checkRange, checkRange):
            checkImg = translateImage(imgH, imgW, image, [0, 0], [xMovement, yMovement] )
            subtractedImg = checkImg - initialDiffImg
            if abs(np.mean(subtractedImg)) < currentMean:
                currentMean = abs(np.mean(subtractedImg))
                xTrans = xMovement
                yTrans = yMovement
    correctedImage = translateImage(imgH, imgW, image, [0, 0], [xTrans, yTrans])
    correctedIntensities = correctedIntensities + [np.array(correctedImage)]
print("no output images = " + str(len(correctedIntensities)))

#correctedDifferences = driftCorrect(differenceImages)
#correctedIntensities = driftCorrect(intensityImages)
plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

img1 = correctedDifferences[0]
img2 = correctedIntensities[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = 100, vmax = 150)
fig2 = ax2.imshow(img2, cmap="gray")


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferences[sb.val]
    img2 = correctedIntensities[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("differences")
ax2.set_title("intensities")
print("plotted")
sb.on_changed(update)
plt.show()
