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
from matplotlib.widgets import RadioButtons

def choseROI(firstImg):
    # Select ROI
    w, h = firstImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    #firstImg = firstImg * 255


    firstImg = np.array(firstImg, dtype = "uint8")


    #firstImg = cv2.convertScaleAbs(firstImg, alpha=0, beta=150)
    checkImg = np.clip(firstImg, 110, 150)
    checkImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    r = cv2.selectROI("select ROI", firstImg)
    #r = cv2.selectROI("select ROI", drawing)
  
    # Crop image to selected reigon
    croppedImage = firstImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    return croppedImage

def templateMatch(img, croppedImg):

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
        print("average loop")
        
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
#imageNumbers = ["321124-321125", "321127-321136", "321227-321236"]

differenceImages = []
intensityImages = []

for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages




angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + 45
intensityImages[0] = rotateImg(intensityImages[0], 45)
differenceImages[0] = rotateImg(differenceImages[0], 45)
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(1, len(intensityImages)):

    #intensityImages[i] = rotateImg(initialIntensityImages[i], angles[i])
    #differenceImages[i] = rotateImg(initialDifferenceImages[i], angles[i])
    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

linePositions = []

for i in range(0, len(differenceImages)):
    print("new image")
    currentLinePositions = []
    for x in range(0, 2):
        plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

    
        img1 = differenceImages[i]
        img2 = intensityImages[i]

        fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
        fig2 = ax2.imshow(img2, cmap="gray", vmin = 5, vmax = 30)

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

        #runs when slider moved
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

        #runs when slider moved
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
    startXVals = np.array(startXVals, dtype = np.float)
    newXVals = np.array(newXVals, dtype = np.float)

    startSize = np.mean([abs(startXVals[0] - startXVals[1]), abs(startXVals[2] - startXVals[3])])
    newSize = np.mean([abs(newXVals[0] - startXVals[1]), abs(newXVals[2] - newXVals[3])])

    
    #diff = abs(diff)

    #sizeX, sizeY = img.shape
    #scaleFactor = np.mean(newXVals / startXVals)
    scaleFactor = startSize / newSize
    print("scale factor = " + str(scaleFactor))
    w, h = initialImg.shape
    newSize = w * scaleFactor
    newSize = round(newSize)


    newImg = cv2.resize(img, (newSize, newSize), fx = scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    w, h = initialImg.shape
    newSizeW, newSizeH = newImg.shape
    diff = (newSizeW - w) / 2

    midw = newSizeW / 2
    midh = newSizeH / 2

    #midw = midw + diff
    #midh = midh + diff
    newImg = newImg[int(round(midw - (w/2))):int(round(midw + (w/2))), int(round(midh - (h/2))):int(round(midh + h/2))]
    #newImg = (newImg-np.nanmin(newImg))/(np.nanmax(newImg)-np.nanmin(newImg))
    #newImg = newImg * 255

    return newImg

initialH, initialW = differenceImages[0].shape
newImg = differenceImages[0]


for i in range(1, len(differenceImages)):
    print("difference")

    startXVals = linePositions[0]
    newXVals = linePositions[i]
    img = differenceImages[i]


    newImg = zoomAdjust(startXVals, newXVals, img, differenceImages[0])

    image = np.array(newImg, dtype = np.float32)

    #image = (newImg + 1)/2
    #image = image * 255
    #image = np.array(image, dtype="uint8")


    differenceImages[i] = image
    print("intensity")
    img = intensityImages[i]

    newImg = zoomAdjust(startXVals, newXVals, img, intensityImages[0])

    np.clip(newImg, 0, 255)
    image = np.array(newImg, dtype = "uint8")

    intensityImages[i] = image

#just rescaled images to line - next need manual shift + final rescale + save individual images
rVals = []
for i in range(0, 3):
    displayImg = np.clip(differenceImages[0], -0.08, 0.1)
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

    # Select ROI
    w, h = displayImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    r = cv2.selectROI("select ROI", displayImg)
    rVals = rVals + [r]


for i in range(0, len(differenceImages) - 1):
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    average = averageImages([differenceImages[0], differenceImages[i+1]])
    ax1.imshow(average, cmap="gray", vmin = -0.08, vmax = 0.1)
    ax2.imshow(differenceImages[i], cmap="gray", vmin = -0.08, vmax = 0.1)
    ax4.imshow(differenceImages[i+1], cmap="gray", vmin = -0.08, vmax = 0.1)

    print("current angles = " + str(angles[i]) + ", and " + str(angles[i+1]))

    currentDifferenceImages = differenceImages[i:i+1]
    #rVals = [[800, 850, 800, 850], [700, 750, 700, 750], [600, 650, 600, 650]]


    for x in range(0, 3):
        reigonIntensities = []

        r = rVals[x]
        print("current r = " + str(r))


        for box in differenceImages:
            # Crop image to selected reigon
            croppedImage = box[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            intensity = np.average(croppedImage)
            reigonIntensities = reigonIntensities + [intensity]

        reigonIntensities = np.array(reigonIntensities)
        reigonIntensities = np.nan_to_num(reigonIntensities)
        print("reigon intensities:")
        print(reigonIntensities)
        # calculate polynomial
        reigonIntensities = np.array(reigonIntensities)
        reigonIntensities = np.nan_to_num(reigonIntensities)
        z = np.polyfit(angles[0:len(reigonIntensities)], reigonIntensities, 3)
        f = np.poly1d(z)

        # calculate new x's and y's
        x_new = np.linspace(np.min(angles), np.max(angles), 50)
        y_new = f(x_new)

        ax3.plot(x_new, y_new)
        ax3.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")
    

    rax = plt.axes([0.47, 0.5, 0.1, 0.15])
    radioHandle = RadioButtons(rax, ("up big", "up", "down big", "down", "big left", "left", "big right", "right", "zoom in", "zoom out"))
    #radioHandle.on_clicked(buttonFunc)
    

    axs = plt.axes([0.15, 0.001, 0.65, 0.03])
    slid = Slider(axs, 'image no', 0, 1, valinit=0, valstep = 1)
    plt.tight_layout()

    #runs when slider moved
    def update(val):
        global differenceImages
        global i
        ax2.clear()
        print("in slider loop")
        images = [differenceImages[0], differenceImages[i+1]]
        #print("img array shape")
        #print(images.shape)
        img1 = images[slid.val]
        print("slider value = " + str(slid.val))

        ax2.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
        plt.draw()
    
    

    imgH, imgW = differenceImages[i].shape
    def buttonFunc(label):
        ax1.clear()
        #ax2.clear()
        ax3.clear()
        global imgH
        global imgW
        global differenceImages
        global i
        global rVals
        global angles

        print("i = " + str(i))

        if label == "up":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [0, 1])
        if label == "up big":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [0, 10])
        if label == "down":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [0, -1])
        if label == "down big":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [0, -10])
        if label == "left":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [-1, 0])
        if label == "right":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [1, 0])
        if label == "big left":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [10, 0])
        if label == "big right":
            differenceImages[i+1] = translateImage(imgH, imgW, differenceImages[i+1], [0, 0], [-10, 0])
        if label == "zoom in":
            newSize = imgH + 1
            differenceImages[i+1] = cv2.resize(differenceImages[i+1], (newSize, newSize), interpolation = cv2.INTER_LINEAR)
            differenceImages[i+1] = differenceImages[i+1][0:imgW, 0:imgH]
        if label == "zoom out":
            newSize = imgH - 1
            differenceImages[i+1] = cv2.resize(differenceImages[i+1], (newSize, newSize), interpolation = cv2.INTER_LINEAR)
            differenceImages[i+1] = differenceImages[i+1][0:imgW, 0:imgH]

        
        
        #currentDifferenceImages = differenceImages[0:i]
        #rVals = [[800, 850, 800, 850], [900, 950, 900, 950], [600, 650, 600, 650]]
        average = averageImages([differenceImages[0], differenceImages[i+1]])
        ax1.imshow(average, cmap="gray", vmin = -0.08, vmax = 0.1)
        #ax2.imshow(differenceImages[0], cmap="gray", vmin = -0.08, vmax = 0.1)
        ax4.imshow(differenceImages[i+1], cmap="gray", vmin = -0.08, vmax = 0.1)

        for x in range(0, 3):
            reigonIntensities = []

            r = rVals[x]
            print("current r = " + str(r))

            for box in differenceImages:
  
                # Crop image to selected reigon
                croppedImage = box[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                intensity = np.average(croppedImage)
                reigonIntensities = reigonIntensities + [intensity]
            reigonIntensities = np.array(reigonIntensities)
            reigonIntensities = np.nan_to_num(reigonIntensities)

            # calculate polynomial
            z = np.polyfit(angles[0:len(reigonIntensities)], reigonIntensities, 3)
            f = np.poly1d(z)

            # calculate new x's and y's
            x_new = np.linspace(np.min(angles), np.max(angles), 50)
            y_new = f(x_new)

            ax3.plot(x_new, y_new)
            ax3.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")
            print("current angles = " + str(angles[i]) + ", and " + str(angles[i+1]))

            plt.draw()
        

    slid.on_changed(update)
    radioHandle.on_clicked(buttonFunc)

    plt.show()

plt.figure()
plt.axis("off")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)

img1 = differenceImages[0]
ax2.plot(angles[0:len(reigonIntensities)], reigonIntensities)

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)


#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")

print("plotted")
sb.on_changed(update)
plt.saveFig("endResult.phg")
plt.show()

averageDifference = averageImages(differenceImages)

plt.imshow(averageDifference, cmap="gray", vmin = -0.08, vmax = 0.1)
plt.show()

for i in range(0, len(differenceImages)):
    sk_imsave("finalCross/alignedCross2_D_%s.tif" %str(i), differenceImages[i])
    sk_imsave("finalCross/alignedCross2_I_%s.tif" %str(i), intensityImages[i])

plt.show()