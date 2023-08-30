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

    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
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


    newImage = initialAvg + ((image - avg) * (initialStd / std))

    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    #image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
    
intensityImages = imageStack
 
for i in range(0, len(differenceImages)):
    image = np.clip(differenceImages[i], 110, 150)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    differenceImages[i] = image * 255


offsetAngle = 50

angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + offsetAngle
intensityImages[0] = rotateImg(intensityImages[0], offsetAngle)
differenceImages[0] = rotateImg(differenceImages[0], offsetAngle)
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(1, len(intensityImages)):

    #intensityImages[i] = rotateImg(initialIntensityImages[i], angles[i])
    #differenceImages[i] = rotateImg(initialDifferenceImages[i], angles[i])
    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

averages = []
boxes = []
for i in range(0, len(differenceImages)):
    # Select ROI
    w, h = differenceImages[i].shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)
    currentBox = choseROI(differenceImages[i])
    currentBox = cv2.resize(currentBox, (1000, 1000), interpolation = cv2.INTER_LINEAR)
    boxes = boxes + [currentBox]
    currentAvg = np.average(currentBox)
    averages = averages + [currentAvg]

plot = plt.figure()

#chose 3 reigons to plot spins
for i in range(0, 1000 - 20):
    reigonIntensities = []
    r = [i, i]

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
