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

    #OpenCV datatype
    firstImg = np.array(firstImg, dtype = "uint8")

    #Adjusting Contrast
    checkImg = np.clip(firstImg, 110, 150)
    checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))

    #Dimentions of user selected box
    r = cv2.selectROI("select ROI", firstImg)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

    #For OpenCV datatype
    img = img * 255
    img = np.array(img, dtype="uint8")

    croppedImg = croppedImg * 255
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
        image = np.array(image, dtype = np.float32)

        #image = cv2.medianBlur(image, 5)
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

    #generates affine translation matrix
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

    #To use centre of image as centre of rotation
    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1

    #Find affine transform for rotation, no rescaling.
    m = cv2.getRotationMatrix2D(center, da, scale)

    #Apply rotation to image
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

#read in images
for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages


#Manually rotating images

angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + 45
intensityImages[0] = rotateImg(intensityImages[0], 45)
differenceImages[0] = rotateImg(differenceImages[0], 45)

for i in range(1, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

#to store positions of user placed line
linePositions = []

#loops through all images
for i in range(0, len(differenceImages)):

    currentLinePositions = []   #just for current image
    for x in range(0, 2):

        plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

    
        img1 = differenceImages[i]
        img2 = intensityImages[i]

        fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
        fig2 = ax2.imshow(img2, cmap="gray", vmin = 5, vmax = 30)

        #for placing intial line
        x1 = 700
        x2 = 900
        y1 = 0
        y2 = 1500
        currentLinePos = x1

        ax1.plot([700, 700], [0, 1500], linewidth = 5, color="red")
        ax2.plot([x2, x2], [y1, y2], linewidth = 5, color="red")
        

        #defines slider axis
        axs = plt.axes([0.15, 0.001, 0.65, 0.03])
        sb = Slider(axs, 'image no', 0, 1500, valinit=0, valstep = 1)

        #runs when slider moved - to find new line positions
        def update(val):
            global currentLinePos
            ax1.clear()
            ax2.clear()
            ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
            ax2.imshow(img2, cmap="gray", vmin = 5, vmax = 30)
            fig1.set_data(img1)
            fig2.set_data(img2)
            x1 = sb.val
            x2 = 900
            y1 = 0
            y2 = 1500

            ax1.plot([x1, x1 - 283], [y1, y2], linewidth = 2, color="red")
            ax2.plot([x1, x1 - 283], [y1, y2], linewidth = 2, color="red")

            currentLinePos = x1

        
            plt.draw()

        ax1.set_title("differences")
        ax2.set_title("intensities")

        sb.on_changed(update)
        plt.show()

        currentLinePositions = currentLinePositions + [currentLinePos]

    #for lines on top and bottom of cross
    for x in range(0, 2):
        plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

    
        img1 = differenceImages[i]
        img2 = intensityImages[i]

        fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
        fig2 = ax2.imshow(img2, cmap="gray", vmin = 5, vmax = 30)

        x1 = 0
        x2 = 1500
        y1 = 800
        y2 = 600
        currentLinePos = x1

        ax1.plot([x1, x2], [y1, y2], linewidth = 2, color="red")
        ax2.plot([x1, x2], [y1, y2], linewidth = 2, color="red")
        

        #defines slider axis
        axs = plt.axes([0.15, 0.001, 0.65, 0.03])
        sb = Slider(axs, 'image no', 0, 1500, valinit=0, valstep = 1)

        #runs when slider moved - to move fit lines
        def update(val):
            global currentLinePos
            ax1.clear()
            ax2.clear()
            ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1 )
            ax2.imshow(img2, cmap="gray", vmin = 5, vmax = 30)
            fig1.set_data(img1)
            fig2.set_data(img2)
            x1 = 0
            x2 = 1500
            y1 = sb.val
            

            ax1.plot([x1, x2], [y1, y1 + 243], linewidth = 2, color="red")
            ax2.plot([x1, x2], [y1, y1 + 243], linewidth = 2, color="red")

            currentLinePos = y1

        
            plt.draw()

        ax1.set_title("differences")
        ax2.set_title("intensities")

        sb.on_changed(update)
        plt.show()
        currentLinePositions = currentLinePositions + [currentLinePos]

    linePositions = linePositions + [currentLinePositions]

def zoomAdjust(startXVals, newXVals, img, initialImg):

    #arrays of line positions
    startXVals = np.array(startXVals, dtype = np.float)
    newXVals = np.array(newXVals, dtype = np.float)

    startSize = np.mean([abs(startXVals[0] - startXVals[1]), abs(startXVals[2] - startXVals[3])])
    newSize = np.mean([abs(newXVals[0] - startXVals[1]), abs(newXVals[2] - newXVals[3])])

    scaleFactor = startSize / newSize
    w, h = initialImg.shape
    newSize = w * scaleFactor
    newSize = round(newSize)

    #Interpolates image to adjust zoom
    newImg = cv2.resize(img, (newSize, newSize), fx = scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    w, h = initialImg.shape
    newImg = newImg[0:w, 0:h]

    return newImg

initialH, initialW = differenceImages[0].shape
newImg = differenceImages[0]

for i in range(1, len(differenceImages)):

    startXVals = linePositions[0]
    newXVals = linePositions[i]
    img = differenceImages[i]

    newImg = zoomAdjust(startXVals, newXVals, img, differenceImages[0])

    image = np.array(newImg, dtype = np.float32)

    differenceImages[i] = image

    img = intensityImages[i]

    newImg = zoomAdjust(startXVals, newXVals, img, intensityImages[0])

    np.clip(newImg, 0, 255)
    image = np.array(newImg, dtype = "uint8")

    intensityImages[i] = image

displayImg = np.clip(differenceImages[i], -0.08, 0.1)
displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))
displayImg = displayImg * 255
plt.imshow(displayImg, cmap="gray")
plt.show()

# Select ROI
w, h = displayImg.shape

# Naming a window
cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

#resizes window, not actual image
cv2.resizeWindow("select ROI", w , h)

r = cv2.selectROI("select ROI", displayImg)
croppedImageD = differenceImages[0][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
print("cropped image chosen")
plt.imshow(croppedImageD, cmap="gray", vmin = -0.08, vmax = 0.1)
plt.show()

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

averageIntensity = averageImages(correctedIntensities)
averageDifference = averageImages(correctedDifferences)

ax4.imshow(averageIntensity, cmap="gray", vmin = 5, vmax = 35)
ax3.imshow(averageDifference, cmap="gray", vmin = -0.08, vmax = 0.1)

sk_imsave("finalCross/Difference.tif", averageDifference)
sk_imsave("finalCross/Intensity.tif", averageIntensity)

for i in range(0, len(differenceImages)):
    sk_imsave("finalCross/alignedCross_D_%s.tif" %str(i), differenceImages[i])
    sk_imsave("finalCross/alignedCross_I_%s.tif" %str(i), intensityImages[i])

plt.show()