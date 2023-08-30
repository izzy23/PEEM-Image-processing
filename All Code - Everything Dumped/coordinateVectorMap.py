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

def templateMatch(img, croppedImg):

    #img = np.array(img, dtype="uint8")
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
        pos = templateMatch(image, croppedImg)
        positions = positions + [[pos[0][0], pos[1][0]]]

    return positions


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

imageNumbers = ["321124-321125", "321127-321136", "321227-321236", "321457-321466", "321527-321536", "321617-321626", "321627-321636"]
#imageNumbers = ["321124-321125", "321127-321136", "321227-321236"]

differenceImages = []
intensityImages = []



for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))
    #plt.imshow(differenceImg, cmap="gray", vmin = -0.08, vmax = 0.1)
    #plt.show()

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

differenceImages = initialDifferenceImages
intensityImages = initialIntensityImages

imageStack = []


initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])



offsetAngle = 50

angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + offsetAngle
#intensityImages[0] = rotateImg(intensityImages[0], offsetAngle)
#differenceImages[0] = rotateImg(differenceImages[0], offsetAngle)ack
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(0, len(intensityImages)):
    print("before rotation")
    plt.imshow(differenceImages[i], cmap = "gray", vmin = -0.08, vmax = 0.1)
    plt.show()

    #intensityImages[i] = rotateImg(initialIntensityImages[i], angles[i])
    #differenceImages[i] = rotateImg(initialDifferenceImages[i], angles[i])
    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])
    print("angle used = " + str(angles[i]))

    print("after rotation")
    plt.imshow(differenceImages[i], cmap = "gray", vmin = -0.08, vmax = 0.1)
    plt.show()

averages = []
boxes = []
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

    img = differenceImages[i]
    currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    currentBox = cv2.resize(currentBox, (1000, 1000), interpolation = cv2.INTER_LINEAR)
    boxes = boxes + [currentBox]
    currentAvg = np.average(currentBox)
    averages = averages + [currentAvg]


#drift correcting cropped square
displayImg = np.clip(boxes[i], -0.08, 0.1)
displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

# Select ROI
w, h = displayImg.shape

# Naming a window
cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

#resizes window, not actual image
cv2.resizeWindow("select ROI", w , h)

r = cv2.selectROI("select ROI", displayImg)
cropp = boxes[0][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

positions = templateMatchStack(boxes, cropp)
imgH, imgW = boxes[0].shape
boxes = translateStack(imgH, imgW, boxes, positions)

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)

img1 = boxes[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = boxes[sb.val]

    fig1.set_data(img1)
        
    plt.draw()


sb.on_changed(update)
plt.show()




plot = plt.figure()

#chose 3 reigons to plot spins
for i in range(0, 3):
    reigonIntensities = []
    displayImg = np.clip(boxes[0], -0.08, 0.1)
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

    # Select ROI
    w, h = displayImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    r = cv2.selectROI("select ROI", displayImg)

    for box in boxes:
  
        # Crop image to selected reigon
        croppedImage = box[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        intensity = np.average(croppedImage)
        reigonIntensities = reigonIntensities + [intensity]
    # calculate polynomial
    z = np.polyfit(angles, reigonIntensities, 3)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(np.min(angles), np.max(angles), 50)
    y_new = f(x_new)

    plt.plot(x_new, y_new)
    plt.plot(angles, reigonIntensities, "*")
plt.show()



plt.plot(averages, angles)
plt.show()
overallAverage = averageImages(boxes)
plt.imshow(overallAverage, cmap="gray")
plt.show()
