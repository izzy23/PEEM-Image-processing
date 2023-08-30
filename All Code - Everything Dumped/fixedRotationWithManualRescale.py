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
        image = np.array(image, dtype = np.float32)
        image = cv2.medianBlur(image, 5)
        image = np.array(image, dtype = np.float32)
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

    image = np.clip(differenceImages[i], -0.05, 0.10)
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

    image = newImage * 255

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
    #image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack


angles = [0, 0, -30, -45, -50, -50, -90]

for i in range(1, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])
    

croppedImage1D1 = choseROI(differenceImages[0])  #select ROI from 1st image in stack
croppedImage1D2 = choseROI(differenceImages[0])
croppedImage1D3 = choseROI(differenceImages[0])
croppedImage1D4 = choseROI(differenceImages[0])
c1 = [croppedImage1D1, croppedImage1D2, croppedImage1D3, croppedImage1D3]
zoom = []

for i in range(1, len(differenceImages)):
    plt.imshow(differenceImages[i], cmap="gray")
    plt.show()

    croppedImage2D1 = choseROI(differenceImages[i])  #select ROI from 1st image in stack
    croppedImage2D2 = choseROI(differenceImages[i])
    croppedImage2D3 = choseROI(differenceImages[i])
    croppedImage2D4 = choseROI(differenceImages[i])
    c2 = [croppedImage2D1, croppedImage2D2, croppedImage2D3, croppedImage2D4]


    for x in range(0, 4):
        zooms = np.array([1, 1, 1, 1], dtype = np.float32)

        plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        fig1 = ax1.imshow(c1[x], cmap="gray")
        fig2 = ax2.imshow(c2[x], cmap="gray")

        #defines slider axis
        axs = plt.axes([0.15, 0.001, 0.65, 0.03])
        sb = Slider(axs, 'zoom', 0.8, 1.4, valinit=1, valstep = 0.01)

        #runs when slider moved
        def update(val):
            global x
            global zooms
            w, h = c1[x].shape
            w2, h2 = c2[x].shape
            img2 = c2[x]
            print("current x = "+str(x))
            scale = sb.val
            img2 = cv2.resize(img2, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
            w, h = img2.shape
            initialW, initialH = c2[x].shape
            img2 = img2[int((w-initialW)//2):int((w+initialH)//2), int((h-initialW)//2):int((h+initialH)//2)]

            #M = np.array([[sb.val, 0], [0, sb.val], [0, 0 ]])
            #M = cv2.getRotationMatrix2D((w2//2,h2//2), None, sb.val)
            #img2 = cv2.warpAffine(c2[x], M, (w,h))


            fig2.set_data(img2)
            zooms[x] = sb.val
        
            plt.draw()

        sb.on_changed(update)
        plt.show()
    zoom = zoom + [np.average(zooms)]

initialW, initialH = differenceImages[0].shape
for i in range(0, len(differenceImages) - 1):
    currentZoom = zoom[i]
    img = cv2.resize(differenceImages[i+1], (0, 0), fx = currentZoom, fy = currentZoom, interpolation = cv2.INTER_LINEAR)
    w, h = img.shape
    img = img[int((w-initialW)//2):int((w+initialH)//2), int((h-initialW)//2):int((h+initialH)//2)]
    differenceImages[i+1] = img

croppedImageD = choseROI(differenceImages[0])  #select ROI from 1st image in stack
#finds positions of selected reigon of interest in all images in stack
positionsD = templateMatchStack(differenceImages, croppedImageD)
#positionsI = templateMatchStack(intensityImages, croppedImageI)

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

averageIntensity = averageImages(intensityImages)
averageDifference = averageImages(differenceImages)

ax4.imshow(averageIntensity, cmap="gray", vmin = 2, vmax =25)
ax3.imshow(averageDifference, cmap = "gray", vmin = -0.1, vmax = 0.1)

for i in range(0, len(correctedDifferences)):
    sk_imsave("rotatedCross/%s_D.tif" % (str(i)), correctedDifferences[i])
    sk_imsave("rotatedCross/%s_I.tif" % (str(i)), correctedIntensities[i])

plt.show()