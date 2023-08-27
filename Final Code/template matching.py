import numpy as np
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave

def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    
    return rotatedImg


def choseROI(firstImg):
    # Select ROI
    w, h = firstImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    #Image displayed to select template - not image used, so no data lost
    #changes contrast so can be seen
    checkImg = firstImg
    checkImg = np.clip(firstImg, 0, 50) #0 = bottom of histogram, 50 = top of histogram
    checkImg = firstImg
    checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    checkImg = checkImg * 255

    #data type neeed by openCV
    checkImg = np.array(checkImg, dtype = "uint8")

    r = cv2.selectROI("select ROI", checkImg)
  
    # Crop original image to selected reigon 
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

    #datatype needed by openCV
    img = np.array(img, dtype="uint8")

    #adjusting contrast
    checkImg = np.clip(croppedImg, 0, 50)
    checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg)) #scales pixel values between 0 and 1
    checkImg = checkImg * 255   #so RGB val


    croppedImg = np.array(checkImg, dtype = "uint8")

    #applys OpenCV's template matching function
    #finds convolution between ROI image and other full image at all positions
    matchMatrix = cv2.matchTemplate(img, croppedImg, cv2.TM_CCOEFF_NORMED)

    #finds positions with max similarity score between cropped image and second image 
    loc = np.where(matchMatrix == np.max(matchMatrix))
    
    #x and y positions in new image where best match
    return(loc[1], loc[0])

def templateMatchStack(imStack, croppedImg):
    positions = []

    #finds x and y shift for all images in stack
    for image in imStack:
        pos = templateMatch(image, croppedImg)
        positions = positions + [[pos[0][0], pos[1][0]]]

    return positions

def translateImage(imgH, imgW, img, fixedPos, driftPos):

    #difference in x and y positions for ROI between images
    dx = driftPos[0] - fixedPos[0]
    dy = driftPos[1] - fixedPos[1]

    #generates translation matrix - Affine transform
    translationMatrix = np.array([
    [1, 0, -dx],
    [0, 1, -dy]
    ], dtype=np.float32)

    #apply transformation matrix
    translatedImage = cv2.warpAffine(src=img, M=translationMatrix, dsize=(imgW, imgH))

    return translatedImage

def translateStack(imgH, imgW, imageStack, positions):

    #first image in stack is unchanged - subsequent images are all matched to this.
    fixedPos = positions[0]
    correctedImages = [imageStack[0]]

    #loops through all remeining images, 
    for i in range(1, len(imageStack)):
        currentImage = imageStack[i]
        #translates images, and addes them to array of aligned images
        correctedImages = correctedImages + [translateImage(imgH, imgW, currentImage, fixedPos, positions[i])]

    return correctedImages

def driftCorrect(imgArr):

    imageStack = np.array(imgArr, dtype=np.float32)
    
    croppedImage = choseROI(imageStack[0])  #select ROI from 1st image in stack

    #finds positions of selected reigon of interest in all images in stack
    positions = templateMatchStack(imageStack, croppedImage)

    #to get image dimentions - all have same h and w here
    imgH = len(imageStack[1][0][:])
    imgW = len(imageStack[1][:][0])

    #aligns images using matched positions found
    correctedStack = translateStack(imgH, imgW, imageStack, positions)

    return correctedStack

def averageImages(imStack):

    imageSum = imStack[0]

    #adding pixel vals for each image
    for image in range(1, len(imStack)):
        
        #can't remove float - 8bit wrap around weird thing
        imageSum = imageSum.astype(float) + imStack[image].astype(float)

    #becomes average
    imageSum = imageSum / len(imStack)

    return imageSum

imageNumbers = np.arange(1, 5)

differenceImages = [io.imread(r"finalCross\alignedCrossFinal3_D_0.tif").astype(np.float32)]
intensityImages = [io.imread(r"finalCross\alignedCrossFinal3_I_0.tif").astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"finalCross\alignedCrossFinal3_D_%s.tif" % str(i))
    intensityImg = io.imread(r"finalCross\alignedCrossFinal3_I_%s.tif" % str(i))

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []
initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])


for image in intensityImages:
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


differenceImages = np.array(differenceImages, dtype=np.float32)
intensityImages = np.array(intensityImages, dtype=np.float32)

angles = [0, 0, 2, 0, 0, 0, 0]
angles = np.array(angles, dtype = np.int)



for i in range(0, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

    initialDifferenceImages[i] = rotateImg(initialDifferenceImages[i], angles[i])
    initialIntensityImages[i] = rotateImg(initialIntensityImages[i], angles[i])
    
croppedImage = choseROI(differenceImages[0])  #select ROI from 1st image in stack

#finds positions of selected reigon of interest in all images in stack
positionsD = templateMatchStack(differenceImages, croppedImage)

imgH = len(differenceImages[1][0][:])
imgW = len(differenceImages[1][:][0])

#shifts untouched images using matched positions
correctedDifferences = translateStack(imgH, imgW, initialDifferenceImages, positionsD)


#finds positions of selected reigon of interest in all images in stack
#will work on intensity images, but less accurate than directly matching the domains.
#semse better to apply the transformation found using difference images to both images.
#positionsI = templateMatchStack(intensityImages, croppedImage)

#alighs images using matched positions found from difference images
correctedIntensities = translateStack(imgH, imgW, initialIntensityImages, positionsD)

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

averageIntensity = averageImages(correctedIntensities)
averageDifference = averageImages(correctedDifferences)

ax4.imshow(averageIntensity, cmap="gray", vmin = 10, vmax = 25)
ax3.imshow(averageDifference, cmap="gray", vmin = -0.08, vmax = 0.1)
print("plotted")
sb.on_changed(update)
plt.show()

#sk_imsave("writeImages/321457-321466_Diffference.tif", averageDifference)
#sk_imsave("writeImages/321457-321466_Intensity.tif", averageIntensity)

for i in range(0, len(correctedDifferences)):
    sk_imsave("finalCross/templateMatched3_D_%s" % str(i), correctedDifferences[i])
    sk_imsave("finalCross/templateMatched3_I_%s" % str(i), correctedIntensities[i])
