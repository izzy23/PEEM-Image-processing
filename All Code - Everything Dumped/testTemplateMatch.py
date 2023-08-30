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
    print(firstImg)

    firstImg = np.array(firstImg, dtype = "uint8")

    #edgeImg = cv2.Canny(image = firstImg, threshold1 = 100, threshold2 = 200)

    #img_blur = cv2.GaussianBlur(firstImg, (3,3), 0) 
    #finds contrast 
    equ = cv2.equalizeHist(firstImg) 

    # attempt debluring
    kernel = np.array([[-1,-1,-1], [-1,50,-1], [-1,-1,-1]])
    img_blur = cv2.filter2D(equ, -1, kernel)

    #img_blur = firstImg
 
    # Sobel Edge Detection
    #sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    #sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    #edgeImg = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    edgeImg = cv2.Canny(image = img_blur, threshold1 =30, threshold2 = 50)


    contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((img_blur.shape[0], img_blur.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)



    #firstImg = cv2.convertScaleAbs(firstImg, alpha=0, beta=150)
    r = cv2.selectROI("select ROI", edgeImg)
    #r = cv2.selectROI("select ROI", drawing)
  
    # Crop image to selected reigon
    #croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    croppedImage = edgeImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

    #img = np.array(img, dtype="uint8")
    img = img * 255
    img = np.array(img, dtype="uint8")
    img = cv2.Canny(image = img, threshold1 = 5, threshold2 = 200)

    #plt.imshow(img, cmap="gray")
    #plt.show()
    #croppedImg = Image.fromarray(croppedImg, "L")

    #removed for edge test
    #croppedImg = croppedImg * 255
    #croppedImg = np.array(croppedImg, dtype="uint8")


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
    #initialImageStack = np.array(imgArr, dtype = np.float32)
    #imageStack = []

    #for image in initialImageStack:
    #    displayMin = 0.00005
    #    displayMax = 0.06
    #    image.clip(displayMin, displayMax, out=image) #hopefully adjust contraast
    #    image = np.array(image, dtype=np.float32)
    #    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    #    imageStack = imageStack + [image]

    imageStack = np.array(imgArr, dtype=np.float32)
    
    croppedImage = choseROI(imageStack[0])  #select ROI from 1st image in stack

    #plt.imshow(imageStack[0], cmap="gray")
    #plt.show()

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

imageNumbers = np.arange(321128, 321137)

differenceImages = [io.imread(r"writeImages\321127_D_avg.tif").astype(np.float32)]
intensityImages = [io.imread(r"writeImages\321127_I_avg.tif").astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"writeImages\%s_D_avg.tif" % str(i))
    intensityImg = io.imread(r"writeImages\%s_I_avg.tif" % str(i))
    print("shape")
    print(differenceImg.shape)
    print(intensityImg.shape)

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

imageStack = []
for image in differenceImages:
        print("differenceMin = " + str(np.nanmin(image)))
        print("differenceMax = " + str(np.nanmax(image)))
        displayMin = -0.15
        displayMax = 0.35
        image.clip(displayMin, displayMax, out=image) #hopefully adjust contraast
        image = np.array(image, dtype=np.float32)
        image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so show

        imageStack = imageStack + [image]
#imageStack = standardizeStack(imageStack)
differenceImages = imageStack
#min = 9999999999
#max = 0
#for image in differenceImages:

#    image = np.array(image, dtype=np.float32)
#
#    currentMin = np.nanmin(image)
#    currentMax = np.nanmax(image)
#    print("current min = " + str(currentMin))
#    print("current max = " + str(currentMax))

#    if currentMin < min:
#         min = currentMin
#    if max > currentMax:
#         max = currentMax

#imageStack = []

#for image in differenceImages:
#     image = (image-min)/(max-min) #renormalises
#     imageStack = imageStack + [image]

#differenceImages = imageStack



#plt.imshow(differenceImages[2])
#plt.title("differenceImages")
#plt.show()
#print("showing intensity")
#plt.imshow(intensityImages[2])
#plt.title("intenisty")
#plt.show()

imageStack = []
for image in intensityImages:
        print("intensity min = " + str(np.nanmin(image)))
        print("intensityMax = " + str(np.nanmax(image)))
        displayMin = 5
        displayMax = 35
        image.clip(displayMin, displayMax, out=image) #hopefully adjust contraast
        image = np.array(image, dtype=np.float32)
        image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
        
        imageStack = imageStack + [image]
 
intensityImages = imageStack
plt.title("intensity after adjust")
plt.imshow(intensityImages[2])
plt.show()




#differenceImages = np.array(differenceImages)
#intensityImages = np.array(intensityImages)

#differenceImages = interpolateImages(differenceImages)
#intensityImages = interpolateImages(intensityImages)

#print("no images after")
#print(len(differenceImages))


print("plotitng")
correctedDifferences = driftCorrect(differenceImages)
correctedIntensities = driftCorrect(intensityImages)
plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

img1 = correctedDifferences[0]
img2 = correctedIntensities[0]

fig1 = ax1.imshow(img1, cmap="gray")
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


