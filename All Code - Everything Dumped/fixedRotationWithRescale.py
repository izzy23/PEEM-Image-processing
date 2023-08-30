import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider


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


    firstImg = np.array(firstImg, dtype = np.float32)

    #checkImg = np.clip(firstImg, 120, 150)     #Worked for all cross in first set
    #ADJUST CONTRAST FOR SELECTION IMAGE HERE :)
    #removed contrast - semed to not be needed here
    #checkImg = np.clip(firstImg, 0, 135)
    checkImg = firstImg


    #checkImg = cv2.equalizeHist(checkImg) 
    #checkImg = np.clip(checkImg, -0.1, 0.1)
    
    #checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    r = cv2.selectROI("select ROI", firstImg)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

    #img = img * 255
    #img = np.array(img, dtype="uint8")


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

def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    return rotatedImg



def averageImages(imStack):
    wMin = 99999999
    hMin = 99999999
    
    for img in imStack:
        w, h = img.shape
        print("found size")
        print(w)
        if w < wMin:
            wMin = w
        if h < hMin:
            hMin = h
    h = hMin
    w = wMin
    print("w = " + str(w))
    print("h = " + str(h))
    centreW = len(imStack[0][:][0]) / 2
    centreH = len(imStack[0][0][:]) / 2

    halfH = h / 2
    #halfH = halfH + 1   #account for weird rouning
    halfW = w / 2

    if centreW - halfW < 0:
        centreW = centreW + 1
    if centreW + halfW > wMin:
        halfW = halfW - 1
    if centreH-halfH < 0:
        centreH = centreH + 1
    if centreH + halfH > hMin:
        halfH = halfH - 1


    #imageSum = np.array(imStack[0][0:w, 0:h], dtype = np.float32)
    #crop around image centre
    imageSum = np.array(imStack[0][int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)])
    print("initial shape")
    print(imageSum.shape)

    #adding pixel vals for each image
    for image in range(1, len(imStack)):
        print("current shape")
        print(img.shape)
        img = imStack[image]

        #img = np.array(img[0:w, 0:h], dtype = np.float32) 
        img = np.array(img[int(centreW-halfW):int(centreW+halfW), int(centreH-halfH):int(centreH+halfH)], dtype = np.float32)

        #can't remove float - 8bit wrap around weird thing
        imageSum = imageSum.astype(float) + img.astype(float)
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

    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) 
    #image = image * 255


    imageStack = imageStack + [image]

correctedDiff = imageStack



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

#angles = [0, 0, -30, -60, -30, -60, -90]   #worked for 0, 1, 2, 6
#angles = [0, 0, -30, -45, -60, -30, -90]
angles = [0, 0, -30, -45, -50, -50, -90]    #best match so far :)
#angles = [0, 0, -30, 180, -15, 0, 0]

for i in range(1, len(intensityImages)):

    intensityImages[i] = rotateImg(initialIntensityImages[i], angles[i])
    #differenceImages[i] = rotateImg(initialDifferenceImages[i], angles[i])   
    differenceImages[i] = rotateImg(correctedDiff[i], angles[i])

croppedImageD = choseROI(differenceImages[0])  #select ROI from 1st image in stack
#croppedImageI = choseROI(intensityImages[0])

initialW, initialH = intensityImages[0].shape

scaleChecks = np.linspace(1, 1.5, 50)
print("no of difference images")
print(len(differenceImages))

for i in range(0, len(differenceImages)):
    zoom = scaleChecks[0]
    currentMax = 0
    img = differenceImages[i]
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
        print("current match score")
        print(np.nanmax(matchMatrix))
        #print(np.nanmax(matchMatrix))
        if np.max(matchMatrix) > currentMax:
            currentMax = np.max(matchMatrix)
            zoom = scale
    img = differenceImages[i]
    print("no of difference images end")
    print(len(differenceImages))

    img = cv2.resize(img, (0, 0), fx = zoom, fy = zoom, interpolation = cv2.INTER_LINEAR)
    h, w = img.shape

    img = img[int((w-initialW)//2):int((w+initialH)//2), int((h-initialW)//2):int((h+initialH)//2)]
    print("size of individual image = " + str(img.shape))
    
    print("zoom used = " + str(zoom))


    differenceImages[i] = img

    img = cv2.resize(intensityImages[i], (0, 0), fx = zoom, fy = zoom, interpolation = cv2.INTER_LINEAR)
    h, w = img.shape

    img = img[int((w-initialW)//2):int((w+initialH)//2), int((h-initialW)//2):int((h+initialH)//2)]



    intensityImages[i] = img
#intensityImages = np.array(intensityImages, dtype = "uint8")
#differenceImages = np.array(differenceImages, dtype = "uint8")


#finds positions of selected reigon of interest in all images in stack
differenceImages = np.array(differenceImages, dtype = np.float32)
croppedImageD = np.array(croppedImageD, dtype = np.float32)
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

sb.on_changed(update)






averageIntensity = averageImages(correctedIntensities)
averageDifference = averageImages(correctedDifferences)

ax4.imshow(averageIntensity, cmap="gray", vmin = 2, vmax =25)
ax3.imshow(averageDifference, cmap = "gray", vmin = -0.1, vmax = 0.1)

plt.show()