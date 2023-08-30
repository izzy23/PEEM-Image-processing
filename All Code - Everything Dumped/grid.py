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


def zoomAdjust(startXVals, newXVals, img, initialImg):
    startXVals = np.array(startXVals, dtype = np.float)
    newXVals = np.array(newXVals, dtype = np.float)

    startSize = np.mean([abs(startXVals[0] - startXVals[1]), abs(startXVals[2] - startXVals[3])])
    newSize = np.mean([abs(newXVals[0] - startXVals[1]), abs(newXVals[2] - newXVals[3])])

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



offset = 50
angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + offset
#intensityImages[0] = rotateImg(intensityImages[0], 45)
#differenceImages[0] = rotateImg(differenceImages[0], 45)
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(0, len(intensityImages)):

    #intensityImages[i] = rotateImg(initialIntensityImages[i], angles[i])
    #differenceImages[i] = rotateImg(initialDifferenceImages[i], angles[i])
    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

initialH, initialW = differenceImages[0].shape
newImg = differenceImages[0]

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

img1 = differenceImages[0]
edgeImg = np.clip(img1, -0.08, 0.1)
edgeImg = (edgeImg-np.nanmin(edgeImg))/(np.nanmax(edgeImg)-np.nanmin(edgeImg))
edgeImg = edgeImg * 255
edgeImg = np.clip(edgeImg, 0, 255)
edgeImg = np.array(edgeImg, dtype="uint8")
#v = np.median(edgeImg)
#sigma = np.nanstd(edgeImg)

#---- apply automatic Canny edge detection using the computed median----
#lower = int(max(0, (1.0 - sigma) * v))
#upper = int(min(255, (1.0 + sigma) * v))

upper = 180
lower = 100
#print("lower = " + str(lower))
#print("upper = " + str(upper))
img2 = cv2.Canny(edgeImg, lower, upper)

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)
fig2 = ax2.imshow(img2, cmap="gray")

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)


#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]
    edgeImg = np.clip(img1, -0.08, 0.1)
    edgeImg = (edgeImg-np.nanmin(edgeImg))/(np.nanmax(edgeImg)-np.nanmin(edgeImg))
    edgeImg = edgeImg * 255
    edgeImg = np.clip(edgeImg, 0, 255)
    edgeImg = np.array(edgeImg, dtype="uint8")
    #v = np.median(edgeImg)
    #sigma = np.nanstd(edgeImg)

    #---- apply automatic Canny edge detection using the computed median----
    #lower = int(max(0, (1.0 - sigma) * v))
    #upper = int(min(255, (1.0 + sigma) * v))

    upper = 180
    lower = 100

    img2 = cv2.Canny(edgeImg, lower, upper)
    print("lower = " + str(lower))
    print("upper = " + str(upper))


    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("differences")

print("plotted")
sb.on_changed(update)

plt.show()

grids = []

for i in range(0, len(differenceImages)):
    displayImg = np.clip(differenceImages[i], -0.08, 0.1)
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

    # Select ROI
    w, h = displayImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    r = cv2.selectROI("select ROI", displayImg)
    cropp = differenceImages[i][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cropp = cv2.resize(cropp, (1500, 1500))
    grids = grids + [cropp]


def draw_grid(image, line_space=20):
    H, W = image.shape
    image[0:H:line_space] = 1
    image[:, 0:W:line_space] = 1

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

img1 = grids[0]
img2 = averageImages(grids)
#draw_grid(img1, 20)

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)

fig2 = ax2.imshow(img2, cmap="gray", vmin=-0.08, vmax = 0.1)

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)


#runs when slider moved
def update(val):
    img1 = grids[sb.val]
    #draw_grid(img1, 20)

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")


sb.on_changed(update)

plt.show()


