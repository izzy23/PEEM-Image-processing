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

def interpolateImages(imgArray):
    print("array length = " + str(len(imgArray)))

    
    interpolatedImages = []

    for currentImg in imgArray:
        
        #img = cv2.imread(currentImg)
        #normalise - so all pixel values less than 1
        currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        
        img = currentImg

        #changed fx and fy = 10 to 2 so maybe quicker for now
        #This gives sub-pixel resolution
        #inter linear is just linear, can switch - LINEAR RECOMENDED BY WEBSITE
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_LINEAR)
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

    #edgeImg = cv2.Canny(image = firstImg, threshold1 = 100, threshold2 = 200)

    #img_blur = cv2.GaussianBlur(firstImg, (3,3), 0) 
    #finds contrast 
    #checkImg = cv2.equalizeHist(firstImg) 

    # attempt debluring
    #kernel = np.array([[-1,-1,-1], [-1,50,-1], [-1,-1,-1]])
    #img = cv2.filter2D(equ, -1, kernel)

    #img_blur = firstImg


    #firstImg = cv2.convertScaleAbs(firstImg, alpha=0, beta=150)
    #checkImg = np.clip(firstImg, 120, 150)     #Worked for all cross in first set
    checkImg = np.clip(firstImg, 100, 150)
    checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    
    r = cv2.selectROI("select ROI", checkImg)
    #r = cv2.selectROI("select ROI", drawing)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

    #img = np.array(img, dtype="uint8")
    img = img * 255
    img = np.array(img, dtype="uint8")


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
         



    #arr = np.array(arr, dtype = np.float32)
    #arr = arr / len(imgStack)
    #mean = np.mean(arr)
    #std = np.std(arr)
    #img = (arr - mean) / std
    return img



def driftCorrect(imgArr):


    imageStack = np.array(imgArr, dtype=np.float32)
    
    croppedImage = choseROI(imageStack[0])  #select ROI from 1st image in stack


    #finds positions of selected reigon of interest in all images in stack
    positions = templateMatchStack(imageStack, croppedImage)

    #imgH, imgW = imageStack[1].shape[:2]
    imgH, imgW = imageStack[1].shape
    imgH = len(imageStack[1][0][:])
    imgW = len(imageStack[1][:][0])



    correctedStack = translateStack(imgH, imgW, imageStack, positions)

    return correctedStack

def averageImages(imStack):

    imageSum = imStack[0]

    #adding pixel vals for each image
    for image in range(1, len(imStack)):
        print("image no = "+str(image))
        
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
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #newImage = avg + ((image - initialAvg) * (std / initialStd))
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    image = cv2.medianBlur(image, 5)

    imageStack = imageStack + [image]
#imageStack = standardizeStack(imageStack)

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



zoomChecks = np.linspace(0.8, 1.5, 50)
positionsD = []
resizedDifferences = np.array([])
resizedIntensities = np.array([])
wMin = 99999999
hMin = 99999999

for i in range(0, len(differenceImages)):
    bestMatch = 0
    img = differenceImages[i]
    img = img * 255
    img = np.array(img, dtype="uint8")

    for scale in zoomChecks:
        resizedImg = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)

        #applys OpenCV's template matching function
        #finds convolution between ROI image and other full image at all positions
        matchMatrix = cv2.matchTemplate(resizedImg, croppedImage, cv2.TM_CCOEFF_NORMED)

        if np.max(matchMatrix) > bestMatch:
            bestMatch = np.max(matchMatrix)

            #finds positions with max similarity score between cropped image and second image 
            bestPos = np.where(matchMatrix == np.max(matchMatrix))
            zoom = scale
    print("zoom = " + str(zoom))

    resizedDiff = cv2.resize(img, (0, 0), fx = zoom, fy = zoom, interpolation = cv2.INTER_LINEAR)
    resizedInt = cv2.resize(intensityImages[i], (0, 0), fx = zoom, fy = zoom, interpolation = cv2.INTER_LINEAR)


    w, h = resizedDiff.shape
    if w < wMin or h < hMin:
        if w < wMin:
            wMin = w
        if h < hMin:
            hMin = h

        centreW = w / 2
        centreH = h / 2

        halfH = h / 2
        halfW = w / 2
    else:
        centreW = wMin / 2
        centreH = hMin / 2

        halfH = hMin / 2
        halfW = wMin / 2
   
    if centreW - halfW < 0:
        centreW = centreW + 1
    if centreW + halfW > wMin:
        halfW = halfW - 1
    if centreH-halfH < 0:
        centreH = centreH + 1
    if centreH + halfH > hMin:
        halfH = halfH - 1

    print("current allowed x range")
    print(str(centreW - halfW) + ", " + str(centreW + halfW))
    print(str(centreH - halfH) + ", " + str(centreH + halfH))
    
    newDifferenceImages = []
    newIntensityImages = []

    #crop around image centre
    for x in range(0, len(resizedDifferences)):

        newDifferenceImages = newDifferenceImages + [resizedDifferences[x][int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)]]
        newIntensityImages = newIntensityImages + [resizedIntensities[x][int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)]]

    if i == 0:
        newDifferenceImages = [resizedDiff[int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)]]
        newIntensityImages = [resizedInt[int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)]]
    else:
        newDifferenceImages = newDifferenceImages + [resizedDiff[int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)]]
        newIntensityImages = newIntensityImages + [resizedInt[int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)]]

    resizedDifferences = newDifferenceImages
    resizedIntensities = newIntensityImages

differenceImages = resizedDifferences
intensityImages = resizedIntensities
print("printing all shapes")
for img in differenceImages:
    print(img.shape)

#print(differenceImages)
imgW, imgH = differenceImages[0].shape

croppedImage = choseROI(differenceImages[0])
#finds positions of selected reigon of interest in all images in stack
positionsD = templateMatchStack(differenceImages, croppedImage)

correctedDifferences = translateStack(imgH, imgW, initialDifferenceImages, positionsD)

#correctedIntensities = translateStack(imgH, imgW, intensityImages, positions)
correctedIntensities = translateStack(imgH, imgW, initialIntensityImages, positionsD)


#correctedDifferences = driftCorrect(differenceImages)
#correctedIntensities = driftCorrect(intensityImages)
plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
#ax3 = plt.subplot(2, 2, 3)
#ax4 = plt.subplot(2, 2, 4)

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

#averageIntensity = averageImages(correctedIntensities)
#averageDifference = averageImages(correctedDifferences)

#ax4.imshow(averageIntensity, cmap="gray", vmin = 2, vmax = 25)
#ax3.imshow(averageDifference, cmap="gray", vmin = -0.1, vmax = 0.1)
print("plotted")
sb.on_changed(update)
plt.show()

#sk_imsave("C254 - Cross Centre/321718-321719_Diffference.tif", averageDifference)
#sk_imsave("C254 - Cross Centre/321718-321719_Intensity.tif", averageIntensity)

