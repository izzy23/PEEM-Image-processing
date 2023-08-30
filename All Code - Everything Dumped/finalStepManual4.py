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
import scipy.optimize as scipy
from skimage.restoration import unwrap_phase



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

crossCentre = np.array([[713, 612], [711, 714], [785, 786], [758, 756], [806, 809], [799, 800], [857, 856]])

imageNumbers = np.arange(1, 7)

#differenceImages = [io.imread(r"finalCross/alignedCrossNewCentreTest22_D_0.tif")]
differenceImages = [io.imread(r"finalCross/alignedCrossNewCentreTest30_D_0.tif")]
intensityImages = [io.imread(r"finalCross/alignedCrossNewCentreTest30_I_0.tif")]

for i in imageNumbers: 

    #differenceImg = io.imread(r"finalCross/alignedCrossNewCentreTest22_D_%s.tif" % str(i))
    intensityImg = io.imread(r"finalCross/alignedCrossNewCentreTest30_I_%s.tif" % str(i))
    differenceImg = io.imread(r"finalCross/alignedCrossNewCentreTest30_D_%s.tif" % str(i))

    print("shape")
    print(differenceImg.shape)
    print(intensityImg.shape)


    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

diffImg = io.imread(r"finalCross/testFirst_D_1.tif")
differenceImages[0] = diffImg


diffImg = io.imread(r"finalCross/testFirst_D_2.tif")
differenceImages[1] = diffImg


print("no of images = " + str(len(differenceImages)))


#offsetAngle = 50
offsetAngle = 0

#angles = [0, 0, -30, -45, -50, -50, -90]
angles = [0, 0, 0, 0, 0, 0, 0]


w, h = differenceImages[0].shape
#crossCentre = [[w/2, h/2], [w/2, h/2], [w/2, h/2], [w/2, h/2], [w/2, h/2], [w/2, h/2], [w/2, h/2]]

angles = np.array(angles, dtype = np.int)
angles = angles + offsetAngle
#intensityImages[0] = rotateImg(intensityImages[0], offsetAngle)
#differenceImages[0] = rotateImg(differenceImages[0], offsetAngle)ack
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(0, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

    intensityImages[i] = np.nan_to_num(intensityImages[i])
    differenceImages[i] = np.nan_to_num(differenceImages[i])

    #differenceImages[i] = unwrap_phase(differenceImages[i])
    
    



img1 = differenceImages[0]
img2 = intensityImages[0]


plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)


img1 = differenceImages[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")

sb.on_changed(update)
plt.show()


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


for i in range(0, len(differenceImages)):
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    ax3.set_yticks([ -0.02, -0.01, 0, 0.01, 0.02])
    unwarpedImg = differenceImages[i]
    unwarpedAverage = averageImages([unwarpedImg, differenceImages[i]])
    average = averageImages([differenceImages[0], differenceImages[i]])
    #average = averageImages([differenceImages[0], differenceImages[i]])
    ax1.imshow(average, cmap="gray", vmin = -0.08, vmax = 0.1)
    ax2.imshow(differenceImages[0], cmap="gray", vmin = -0.08, vmax = 0.1)
    #ax3.imshow(unwarpedAverage, cmap="gray", vmin = -0.08, vmax = 0.1)
    ax4.imshow(differenceImages[i], cmap="gray", vmin = -0.08, vmax = 0.1)

    print("current angles = " + str(angles[i]))

    #currentDifferenceImages = differenceImages[i:i+1]
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
        #z = np.polyfit(angles[0:len(reigonIntensities)], reigonIntensities, 3)
        #f = np.poly1d(z)

        # calculate new x's and y's
        #oldAngles = np.array([0, 0, -26, -45, -49, -49, -90])
        oldAngles = np.array([50, 50, 80, 100, 110, 110, 140])
        angles = np.array(angles)
        displayAngles = angles + oldAngles
        #x_new = np.linspace(np.min(displayAngles), np.max(displayAngles), 50)
        #y_new = f(x_new)
        offset = 20
        #ax3.plot(x_new, y_new)
        ax3.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
        ax3.plot(displayAngles[0:len(reigonIntensities)], reigonIntensities, "*")
        ax3.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
        x_new = np.linspace(np.min(displayAngles), np.max(displayAngles), 100)
        x_new_rad = np.deg2rad(x_new + offset)
        y_new = 0.02 * np.sin(2 * x_new_rad)

        ax3.plot(x_new, y_new)
        ax3.plot(displayAngles[0:len(reigonIntensities)], reigonIntensities, "*")
        ax3.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])

        xVals = np.linspace(-50, 0, 100)
        #xValsIn = ((xVals + 30) * np.pi) / 180
        #yVals = np.sin(2 * xValsIn)
        #yVals = yVals * 0.02
        #ax3.plot(xVals, yVals, "-")
    

    rax = plt.axes([0.47, 0.5, 0.1, 0.15])
    radioHandle = RadioButtons(rax, ("up big", "up", "down big", "down", "big left", "left", "big right", "right", "zoom in", "zoom out", "+ rotate", "- rotate"))
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
        images = [differenceImages[0], differenceImages[i]]
        #print("img array shape")
        #print(images.shape)
        img1 = images[slid.val]
        print("slider value = " + str(slid.val))

        ax2.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
        plt.draw()
    
    

    imgH, imgW = differenceImages[i].shape
    def buttonFunc(label):
        
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        global imgH
        global imgW
        global differenceImages
        global i
        global rVals
        global angles
        global unwarpedImg
        global crossCentre

        print("centre x = " + str(crossCentre[i][0]))
        print("centre y = " + str(crossCentre[i][1]))
        crossCentre = np.round(crossCentre)
        crossCentre = np.array(crossCentre, dtype = np.float32)

        print("i = " + str(i))
        ax3.set_yticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
        

        if label == "up":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [0, 1])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [0, 1])
            crossCentre[i][1] = crossCentre[i][0] + 1 
        if label == "up big":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [0, 5])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [0, 5])
            crossCentre[i][1] = crossCentre[i][0] + 5
        if label == "down":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [0, -1])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [0, -1])
            crossCentre[i][1] = crossCentre[i][0] - 1
        if label == "down big":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [0, -5])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [0, -5])
            crossCentre[i][1] = crossCentre[i][0] - 5
        if label == "left":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [1, 0])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [-1, 0])
            crossCentre[i][1] = crossCentre[i][1] - 1
        if label == "right":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [-1, 0])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [1, 0])
            crossCentre[i][1] = crossCentre[i][1] + 1
        if label == "big left":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [5, 0])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [5, 0])
            crossCentre[i][1] = crossCentre[i][1] + 5
        if label == "big right":
            differenceImages[i] = translateImage(imgH, imgW, differenceImages[i], [0, 0], [-5, 0])
            unwarpedImg = translateImage(imgH, imgW, unwarpedImg, [0, 0], [-5, 0])
            crossCentre[i][1] = crossCentre[i][1] - 5
        if label == "zoom in":

            print("centre x = " + str(crossCentre[i][0]))
            print("centre y = " + str(crossCentre[i][1]))
            m = cv2.getRotationMatrix2D([crossCentre[i][0], crossCentre[i][1]], 0, 1.005)
            newImg = cv2.warpAffine(differenceImages[i], m, (w,h), flags = cv2.INTER_CUBIC)
            differenceImages[i] = newImg
            crossCentre[i][:] = ((crossCentre[i][:] * 1.005) + (crossCentre[i][:])) / 2

        if label == "zoom out":
            #m = cv2.getRotationMatrix2D([imgH/2, imgW / 2], 0, 0.99)
            m = cv2.getRotationMatrix2D([crossCentre[i][0], crossCentre[i][1]], 0, 0.995)
            newImg = cv2.warpAffine(differenceImages[i], m, (w,h), flags = cv2.INTER_CUBIC)
            differenceImages[i] = newImg
            crossCentre[i][:] = ((crossCentre[i][:] * 0.995) + (crossCentre[i][:])) / 2
            #differenceImages[i+1] = newImg
        if label == "+ rotate":
            m = cv2.getRotationMatrix2D([imgH/2, imgW / 2], 0.1, 1)
            newImg = cv2.warpAffine(differenceImages[i], m, (w,h), flags = cv2.INTER_CUBIC)
            differenceImages[i] = newImg
            angles[i] = angles[i] + 0.1
        if label == "- rotate":
            m = cv2.getRotationMatrix2D([imgH/2, imgW / 2], -0.1, 1)
            newImg = cv2.warpAffine(differenceImages[i], m, (w,h), flags = cv2.INTER_CUBIC)
            differenceImages[i] = newImg
            angles[i] = angles[i] - 0.1

        
        
        #currentDifferenceImages = differenceImages[0:i]
        #rVals = [[800, 850, 800, 850], [900, 950, 900, 950], [600, 650, 600, 650]]
        averageSame = averageImages([unwarpedImg, differenceImages[i]])
        average = averageImages([differenceImages[0], differenceImages[i]])
        ax1.imshow(average, cmap="gray", vmin = -0.08, vmax = 0.1)
        ax2.imshow(differenceImages[0], cmap="gray", vmin = -0.08, vmax = 0.1)
        #ax3.imshow(unwarpedAverage, cmap="gray", vmin = -0.08, vmax = 0.1)
        ax4.imshow(differenceImages[i], cmap="gray", vmin = -0.08, vmax = 0.1)

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
            print("current intensities")
            print(reigonIntensities)

            # calculate polynomial
            #z = np.polyfit(angles[0:len(reigonIntensities)], reigonIntensities, 3)
            #f = np.poly1d(z)

            # calculate new x's and y's
            offset = 20
            #oldAngles = np.array([0, 0, -26, -45, -49, -49, -90])
            oldAngles = np.array([50, 50, 80, 100, 110, 110, 140])
            angles = np.array(angles)
            displayAngles = angles + oldAngles
            x_new = np.linspace(np.min(displayAngles), np.max(displayAngles), 100)
            x_new_rad = np.deg2rad(x_new + offset)
            y_new = 0.02 * np.sin(2 * x_new_rad)

            ax3.plot(x_new, y_new)
            ax3.plot(displayAngles[0:len(reigonIntensities)], reigonIntensities, "*")
            ax3.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
            #xVals = np.linspace(-50, 0, 100)
            #yVals = np.sin(xVals)
            #ax3.plot(xVals, yVals, "-")
    
            
            print("current angles = " + str(displayAngles[i]))

            plt.draw()
        

    slid.on_changed(update)
    radioHandle.on_clicked(buttonFunc)

    plt.show()

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    ax1.imshow(unwarpedImg, cmap="gray", vmin = -0.08, vmax = 0.1)
    ax2.imshow(differenceImages[i], cmap="gray", vmin = -0.08, vmax = 0.1)

    plt.show()

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)



img1 = differenceImages[0]
#oldAngles = np.array([0, 0, -26, -45, -49, -49, -90])
oldAngles = np.array([50, 50, 80, 100, 110, 110, 140])
#oldAngles = np.array([-50, -50, -26, -5, 1, 1, 45])
angles = np.array(angles)
displayAngles = angles + oldAngles
#x_new = np.linspace(np.min(displayAngles), np.max(displayAngles), 50)
#y_new = f(x_new)

#ax3.plot(x_new, y_new)


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

plt.show()

averageDifference = averageImages(differenceImages)

plt.imshow(averageDifference, cmap="gray", vmin = -0.08, vmax = 0.1)
plt.show()

#for i in range(0, len(differenceImages)):
#    sk_imsave("finalCross/alignedCrossNewCentreTest31_D_%s.tif" %str(i), differenceImages[i])
#    sk_imsave("finalCross/alignedCrossNewCentreTest31_I_%s.tif" %str(i), intensityImages[i])

#plt.show()

print("updated centre positions")
print(crossCentre)
print("final rotation angles")
print(displayAngles)

