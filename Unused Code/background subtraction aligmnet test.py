import numpy as np
import h5py
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io

#attempt at subtracting images to check similarity

def translateImage(imgH, imgW, img, fixedPos, driftPos):

    #difference in x and y positions for ROI between images
    tx = driftPos[0] - fixedPos[0]
    ty = driftPos[1] - fixedPos[1]

    #generates translation matrix
    translationMatrix = np.array([
    [1, 0, -tx],
    [0, 1, -ty]
    ], dtype=np.float32)

    #apply Affine transformation
    translatedImage = cv2.warpAffine(src=img, M=translationMatrix, dsize=(imgW, imgH))

    return translatedImage

def translateStack(imgH, imgW, imageStack, positions):

    fixedPos = positions[0]
    correctedImages = [imageStack[0]]

    for i in range(1, len(imageStack)):
        currentImage = imageStack[i]
        correctedImages = correctedImages + [translateImage(imgH, imgW, currentImage, fixedPos, positions[i])]

    return correctedImages


#reads in a stack of images
imageNumbers = np.arange(321128, 321137)

differenceImages = [io.imread(r"writeImages\321127_D_avg.tif").astype(np.float32)]
intensityImages = [io.imread(r"writeImages\321127_I_avg.tif").astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"writeImages\%s_D_avg.tif" % str(i))
    intensityImg = io.imread(r"writeImages\%s_I_avg.tif" % str(i))

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

imageStack = []

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    #needs to be nanmean otherwise breaks bc dead pixels stored as NaN
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #adjusting contrast - so all images have same mean and standard deviation as 1t image
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")

    #applies blur filter to reduce effect of dead pixels
    image = cv2.medianBlur(image,3)

    #store image
    imageStack = imageStack + [image]

differenceImages = imageStack

initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []

for image in intensityImages:
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #adjusting contrast so all images have same mean and standard deviation as initial
    image = np.clip(image, 0, 30)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))


    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    
    #median blur filter to reduce effects of dead pixels
    image = cv2.medianBlur(image,3)
        
    imageStack = imageStack + [image]
 
intensityImages = imageStack

imgH = len(differenceImages[1][0][:])
imgW = len(differenceImages[1][:][0])

#square size for range of x and y shifts looked for
checkRange = 10
currentMean = 999999999

#first image - used to align all later images
initialDiffImg = differenceImages[2][:][:]

#stores aligned images
correctedDifferences = []

for image in differenceImages:
    #loops through check square of x and y shifts
    for xMovement in range(-checkRange, checkRange):
        for yMovement in range(-checkRange, checkRange):
            #shifts image
            checkImg = translateImage(imgH, imgW, image, [0, 0], [xMovement, yMovement] )
            #subtracts shifted image, to compare similarities
            subtractedImg = checkImg - initialDiffImg

            #if aligned and same contrast, then should have low average pixel value
            if abs(np.mean(subtractedImg)) < currentMean:
                #stores xy shift if new low
                currentMean = abs(np.mean(subtractedImg))
                xTrans = xMovement
                yTrans = yMovement
    
    #translates based on lowest subtraction output
    correctedImage = translateImage(imgH, imgW, image, [0, 0], [xTrans, yTrans])
    correctedDifferences = correctedDifferences + [correctedImage]


#same process as above, but on intensity images instead
currentMean = 999999999
correctedIntensities = []
initialIntensityImg = intensityImages[0]

for image in intensityImages:
    #loops through check square of x and y shifts
    for xMovement in range(-checkRange, checkRange):
        for yMovement in range(-checkRange, checkRange):
            #shifts image
            checkImg = translateImage(imgH, imgW, image, [0, 0], [xMovement, yMovement] )
            #subtracts shifted image to check for similarities
            subtractedImg = checkImg - initialDiffImg

            #if aligned and same contrast, should have low average pixel value
            if abs(np.mean(subtractedImg)) < currentMean:
                #stores xy shift if new low
                currentMean = abs(np.mean(subtractedImg))
                xTrans = xMovement
                yTrans = yMovement

    #translates based on lowest subtraction output
    correctedImage = translateImage(imgH, imgW, image, [0, 0], [xTrans, yTrans])
    correctedIntensities = correctedIntensities + [np.array(correctedImage)]

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

sb.on_changed(update)
plt.show()
