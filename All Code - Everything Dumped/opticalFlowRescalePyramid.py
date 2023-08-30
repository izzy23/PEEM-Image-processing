import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
#from skimage import io
#from skimage.io import imsave as sk_imsave
from skimage.transform import pyramid_gaussian
from scipy.interpolate import Rbf

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
def align_images_tps(image1, image2, num_scales=10):
    aligned_image = None

    # Convert images to the required data type (8-bit unsigned integer)
    image1 = (image1 * 255).astype(np.uint8)
    image2 = (image2 * 255).astype(np.uint8)

    image_shape = image1.shape[::-1]  # Reverse the shape to (height, width)
    pyramid1 = list(pyramid_gaussian(image1, max_layer=num_scales*2-1, sigma=1, mode='constant', multichannel=False, preserve_range=True))
    pyramid2 = list(pyramid_gaussian(image2, max_layer=num_scales*2-1, sigma=1, mode='constant', multichannel=False, preserve_range=True))
    pyramid1 = [cv2.resize(pyramid1[level], image_shape, interpolation=cv2.INTER_NEAREST) for level in range(num_scales*2-1)]
    pyramid2 = [cv2.resize(pyramid2[level], image_shape, interpolation=cv2.INTER_NEAREST) for level in range(num_scales*2-1)]

    # Initialize transformation matrix
    M = np.eye(2, 3)

    # Parameters for feature point detection
    feature_params = dict(maxCorners=800, qualityLevel=0.01, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(51, 51), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Variable to track if feature points are found
    feature_points_found = False

    # Iterate through different scales
    for level in reversed(range(num_scales)):
        # Resize image2 to the current level

        scaled_image2 = pyramid2[level]

        # Detect feature points in both images
        p0 = cv2.goodFeaturesToTrack(np.array(pyramid1[level], dtype=np.float32), mask=None, **feature_params)
        if p0 is None:
            # No feature points found at this scale level
            continue

        p0 = p0.astype(np.float32)

        plt.imshow(pyramid1[level], cmap="gray")
        plt.show()

        plt.imshow(scaled_image2, cmap="gray")
        plt.show()

        p1, st, err = cv2.calcOpticalFlowPyrLK(pyramid1[level], scaled_image2, p0, None, **lk_params)

        # Filter out invalid points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        print("Level:", level)
        print("Number of feature points:", len(good_new))
        print("good_new shape:", good_new.shape)
        print("good_old shape:", good_old.shape)

        # Calculate the transformation matrix between image1 and the scaled_image2
        M = cv2.estimateAffinePartial2D(good_old, good_new, M)[0]

        print("M matrix:")
        print(M)

        # Update the transformation matrix for the next level
        M[:, 2] *= 2

        # Calculate the Thin-Plate Spline transformation
        tps = Rbf(good_old[:, 0], good_old[:, 1], good_new[:, 0], good_new[:, 1])

        # Apply the Thin-Plate Spline transformation to align the images
        aligned_image = tps(*np.meshgrid(np.arange(image2.shape[1]), np.arange(image2.shape[0]))[::-1], indexing='ij')

        # Feature points found at this scale level, set the flag to True
        feature_points_found = True
        break  # Break out of the loop if successful alignment is achieved

    # Check if any feature points were found at any scale level
    if not feature_points_found:
        raise ValueError("Failed to align the images. No feature points found at any scale level.")

    return aligned_image





def align_images_lucas_kanade(image1, image2, num_scales=10):
    image1 = np.array(image1, dtype = np.float32)
    image2 = np.array(image2, dtype = np.float32)
    # Create image pyramids
    pyramid1 = list(pyramid_gaussian(image1, max_layer=num_scales-1))
    pyramid2 = list(pyramid_gaussian(image2, max_layer=num_scales-1))

    # Initialize transformation matrix
    M = np.eye(2, 3)

    # Parameters for feature point detection
    feature_params = dict(maxCorners=800, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Iterate through different scales
    for level in reversed(range(num_scales)):
        # Resize image2 to the current level
        scaled_image2 = pyramid2[level]

        # Detect feature points in both images
        p0 = cv2.goodFeaturesToTrack(pyramid1[level], mask=None, **feature_params)
        if p0 is None:
            break
        p0 = p0.astype(np.float32)

        p1, st, err = cv2.calcOpticalFlowPyrLK(pyramid1[level], scaled_image2, p0, None, **lk_params)

        # Filter out invalid points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        print("points found")
        print(good_new)
        print(good_old)

        # Calculate the transformation matrix between image1 and the scaled_image2
        M = cv2.estimateAffinePartial2D(good_old, good_new, M)[0]

        # Update the transformation matrix for the next level
        M[:, 2] *= 2

    # Warp image2 to match image1
    print("chosen M")
    print(M)
    aligned_image = cv2.warpAffine(image2, M, (image2.shape[1], image2.shape[0]))

    return aligned_image


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
    #image = cv2.medianBlur(image, 5)
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
    print("pre alignment image shape")
    print(img.shape)
    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()
    #differenceImages[i] = align_images_lucas_kanade(firstImg, img)
    differenceImages[i] = align_images_tps(firstImg, img)

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


