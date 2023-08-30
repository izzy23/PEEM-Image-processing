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


def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    return rotatedImg

def align_images_ecc(image1, image2):

    
    # Define warp modes to try (can be cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, or cv2.MOTION_HOMOGRAPHY)
    warp_modes = [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE]

    # Define the number of iterations (increase for better alignment but longer computation time)
    number_of_iterations = 1000

    # Specify the termination criteria for the ECC algorithm (smaller epsilon for better alignment)
    termination_eps = 1e-5

    # Initialize the warp matrix as a 2x3 floating-point matrix filled with zeros
    warp_matrix = np.zeros((2, 3), dtype=np.float32)

    for warp_mode in warp_modes:
        # Find the transformation matrix (warp matrix) using ECC
        _, warp_matrix = cv2.findTransformECC(image1, image2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps))

        # Check if the warp matrix contains NaN values
        if np.isnan(warp_matrix).any():
            continue

        # Warp the input image using the warp matrix
        rows, cols, _ = image2.shape
        aligned_image = cv2.warpAffine(image1, warp_matrix, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return aligned_image

    # If none of the warp modes produced a valid result, return the original image1
    return image1


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

    
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    #print(newImage)
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    image = cv2.medianBlur(image, 5)
    image = np.clip(image, 120, 140)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255
    image = np.array(image, dtype="uint8")

    #plt.imshow(image, cmap="gray")
    #plt.show()

    imageStack = imageStack + [image]
#imageStack = standardizeStack(imageStack)

differenceImages = imageStack
print("no of images")
print(len(differenceImages))

initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []


for image in intensityImages:
    print("looping")
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

angles = [0, 0, -30, -45, -50, -50, -90]

for i in range(1, len(intensityImages)):
    print("before rotate")
    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()


firstImg = differenceImages[0]
for i in range(1, len(differenceImages)):
    img = differenceImages[i]
    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()
    differenceImages[i] = align_images_ecc(img, firstImg)

    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()

correctedDifferences = differenceImages
correctedIntensities = intensityImages

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

averageIntensity = averageImages(intensityImages)
averageDifference = averageImages(differenceImages)

ax4.imshow(averageIntensity, cmap="gray")
ax3.imshow(averageDifference, cmap = "gray")

#for i in range(0, len(correctedDifferences)):
    #sk_imsave("rotatedCross/%s_D.tif" % (str(i)), correctedDifferences[i])
    #sk_imsave("rotatedCross/%s_I.tif" % (str(i)), correctedIntensities[i])

plt.show()


