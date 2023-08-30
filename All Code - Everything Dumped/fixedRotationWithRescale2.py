import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave


def choseROI(firstImg):
    # Select ROI
    w, h = firstImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    #firstImg = firstImg * 255

    print("img max")
    print(np.nanmax(firstImg))


    firstImg = np.array(firstImg, dtype = "uint8")

    checkImg = firstImg

    #checkImg = cv2.equalizeHist(checkImg) 
    
    #checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    r = cv2.selectROI("select ROI", checkImg)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

    img = img * 255
    img = np.clip(img, 0, 255)
    img = np.array(img, dtype="uint8")
    croppedImg = np.clip(croppedImg, 0, 255)
    croppedImg = np.array(croppedImg, dtype="uint8")
    


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
    #translatedImage = (translatedImage + 1)/2
    #translatedImage = translatedImage * 255

    return translatedImage

def translateStack(imgH, imgW, imageStack, positions):
    fixedPos = positions[0]
    correctedImages = [imageStack[0]]

    for i in range(1, len(imageStack)):
        currentImage = imageStack[i]
        correctedImages = correctedImages + [translateImage(imgH, imgW, currentImage, fixedPos, positions[i])]

    return correctedImages


def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    return rotatedImg



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

#array for all images
#imageNumbers = ["321124-321125", "321127-321136", "321457-321466", "321467-321476", "321527-321536", "321537-321546", "321617-321626", "321627-321636"]

#array for images with domains
imageNumbers = ["321124-321125", "321127-321136", "321227-321236", "321457-321466", "321527-321536", "321617-321626", "321627-321636"]
differenceImages = []
intensityImages = []

for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

for i in range(0, len(differenceImages)):

    image = np.clip(differenceImages[i], -0.1, 0.10)
    differenceImages[i] = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) 

imageStack = []

startDiff = differenceImages[0]
#startDiff = startDiff * 255
startDiff = np.array(startDiff, dtype = np.float32)
#startDiff = np.clip(startDiff, 0, 255)
startDiff = cv2.medianBlur(startDiff, 5)
#initialAvg = np.nanmean(differenceImages[0])
#initialStd = np.nanstd(differenceImages[0])np.clip
initialAvg = np.nanmean(startDiff)
initialStd = np.nanstd(startDiff)

for image in differenceImages:
    #image = image * 255
    #image = np.clip(image, 0, 255)
    image = np.array(image, dtype=np.float32)
    image = cv2.medianBlur(image, 5)

    avg = np.nanmean(image)
    std = np.nanstd(image)
    
    newImage = initialAvg + ((image - avg) * (initialStd / std))

    #plt.imshow(newImage, cmap="gray")
    #plt.show()
    #image = (newImage + 1) / 2

    #plt.imshow(newImage, cmap="gray")
    #plt.show()
    #image = np.clip(newImage, -0.03, 0.1)
    #image = np.clip(newImage, -0.07, 0.1)
    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) 
    #image = (newImage-np.nanmin(newImage))/(np.nanmax(newImage)-np.nanmin(newImage)) 
    #image = newImage
    #image = (image + 1)/2
    image = image * 255

    image = np.clip(image, 0, 255)
    


    image = np.array(image, dtype="uint8")

    #image = cv2.medianBlur(image, 5)

    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) 
    #image = image * 255


    imageStack = imageStack + [image]

correctedDiff = imageStack
differenceImages = imageStack



initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []

for image in intensityImages:
    print("looping")
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    #image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))

    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack

print("ROTATIONS")

angles = [0, 0, -30, -45, -50, -50, -90]


for i in range(1, len(intensityImages)):
    print("rotating")

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])
    plt.imshow(differenceImages[i], cmap="gray")
    plt.show()

croppedImageD = choseROI(intensityImages[0])  #select ROI from 1st image in stack


#positionsI = templateMatchStack(intensityImages, croppedImageI)

imgH = len(differenceImages[1][0][:])
imgW = len(differenceImages[1][:][0])

#correctedDifferences = translateStack(imgH, imgW, differenceImages, positionsD)
#correctedIntensities = translateStack(imgH, imgW, intensityImages, positionsD)

initialW = int(imgW * 0.9)
initialH = int(imgH * 0.9)
scaleChecks = np.linspace(0.9, 1.3, 100)

for i in range(0, len(differenceImages)):
    print("in scale check loop")
    zoom = scaleChecks[0]
    currentMax = 0
    img = intensityImages[i]
    #img = differenceImages[i]

    print("in loop")
    print(np.nanmax(img))
    print(np.nanmin(img))
    for scale in scaleChecks:

        resizedTemplate = cv2.resize(croppedImageD, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        #img = (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)) #renormalises - after contrast adjust so shows
        #img = img * 255
        img = np.array(img, dtype = np.float32)
        #resizedTemplate = (resizedTemplate-np.nanmin(resizedTemplate))/(np.nanmax(resizedTemplate)-np.nanmin(resizedTemplate)) #renormalises - after contrast adjust so shows
        #resizedTemplate = resizedTemplate * 255

        #img = np.array(img, dtype="uint8")

        resizedTemplate = np.array(resizedTemplate, dtype = np.float32)

        #img = cv2.Canny(img, 100, 150)
        #resizedTemplate = cv2.Canny(resizedTemplate, 100, 150)

        #applys OpenCV's template matching function
        #finds convolution between ROI image and other full image at all positions
        matchMatrix = cv2.matchTemplate(img, resizedTemplate, cv2.TM_CCOEFF_NORMED)
        #print("current match score")
        #print(np.nanmax(matchMatrix))
        #print(np.nanmax(matchMatrix))
        if np.max(matchMatrix) > currentMax:
            currentMax = np.max(matchMatrix)
            zoom = scale
            print("new zoom found")
            print(zoom)
            print("similarity")
            print(np.nanmax(matchMatrix))
    img = differenceImages[i]
    print("no of difference images end")
    print(len(differenceImages))
    print("just before zoom")
    #plt.imshow(img, cmap="gray")
    #plt.show()
    img = cv2.resize(img, (0, 0), fx = zoom, fy = zoom, interpolation = cv2.INTER_LINEAR)
    h, w = img.shape

    img = img[int((w-initialW)//2):int((w+initialH)//2), int((h-initialW)//2):int((h+initialH)//2)]
    print("final images")
    #plt.imshow(img, cmap="gray")
    #plt.show()
    print("size of individual image = " + str(img.shape))
    
    print("zoom used = " + str(zoom))
    #plt.imshow(img, cmap="gray")
    #plt.show()

    differenceImages[i] = img

    img = cv2.resize(intensityImages[i], (0, 0), fx = zoom, fy = zoom, interpolation = cv2.INTER_LINEAR)
    h, w = img.shape

    img = img[int((w-initialW)//2):int((w+initialH)//2), int((h-initialW)//2):int((h+initialH)//2)]



    intensityImages[i] = img
#intensityImages = np.array(intensityImages, dtype = "uint8")
#differenceImages = np.array(differenceImages, dtype = "uint8")
croppedImageD = choseROI(differenceImages[0]) 
croppedImageD = np.array(croppedImageD, np.float32)


#finds positions of selected reigon of interest in all images in stack
positionsD = templateMatchStack(differenceImages, croppedImageD)

#finds positions of selected reigon of interest in all images in stack
differenceImages = np.array(differenceImages, dtype = np.float32)
croppedImageD = np.array(croppedImageD, dtype = np.float32)
print("difference Images")
print(differenceImages)
print("cropped image")
print(croppedImageD)
positionsD = templateMatchStack(differenceImages, croppedImageD)

imgH = len(differenceImages[1][0][:])
imgW = len(differenceImages[1][:][0])

correctedDifferences = translateStack(imgH, imgW, differenceImages, positionsD)
correctedIntensities = translateStack(imgH, imgW, intensityImages, positionsD)






plt.figure()
plt.axis("off")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

img1 = correctedDifferences[0]
img2 = correctedIntensities[0]

#fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.1, vmax = 0.1)
#fig2 = ax2.imshow(img2, cmap="gray", vmin = 2, vmax = 25)

fig1 = ax1.imshow(img1, cmap="gray")
fig2 = ax2.imshow(img2, cmap="gray")


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

sb.on_changed(update)

averageIntensity = averageImages(intensityImages)
averageDifference = averageImages(differenceImages)

ax4.imshow(averageIntensity, cmap="gray", vmin = 2, vmax =25)
ax3.imshow(averageDifference, cmap = "gray")

#for i in range(0, len(correctedDifferences)):
#    sk_imsave("rotatedCross/%s_D.tif" % (str(i)), correctedDifferences[i])
#    sk_imsave("rotatedCross/%s_I.tif" % (str(i)), correctedIntensities[i])

plt.show()