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
def align_images_multi_scale(image1, image2, num_scales=10):
    def phase_correlation_align(image1, image2):
        # Convert images to grayscale and normalize
        gray_image1 = image1.astype(float) / 255
        gray_image2 = image2.astype(float) / 255

        # Compute the 2D Discrete Fourier Transform
        f1 = np.fft.fft2(gray_image1)
        f2 = np.fft.fft2(gray_image2)

        # Compute the cross-power spectrum
        cross_power_spectrum = f1 * np.conj(f2)

        # Compute the phase correlation, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            phase_correlation = cross_power_spectrum / np.abs(cross_power_spectrum)

        # Handle division by zero and NaN values
        phase_correlation[np.isnan(phase_correlation)] = 0
        phase_correlation[np.isinf(phase_correlation)] = 0

        # Compute the inverse Fourier Transform
        inverse_phase_corr = np.fft.ifft2(phase_correlation)

        # Find the translation that maximizes the correlation
        translation = np.unravel_index(np.argmax(np.abs(inverse_phase_corr)), inverse_phase_corr.shape)
        print("translation found")
        print(translation)

        return translation

    # Ensure both images have the same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Create a copy of image2 for alignment
    aligned_image = image2.copy()

    # Iterate through different scales
    for scale in reversed(range(num_scales)):
        scale_factor = 1 / (2 ** scale)
        print("current scale factor = " + str(scale_factor))

        # Resize the aligned_image
        interpolation = cv2.INTER_LINEAR
        scaled_image2 = cv2.resize(aligned_image, image1.shape, fx=scale_factor, fy=scale_factor, interpolation=interpolation)

        # Find the translation for the current scale using phase correlation
        translation = phase_correlation_align(image1, scaled_image2)

        # Apply translation to the original image2
        M = np.float32([[1, 0, -translation[1]], [0, 1, -translation[0]]])
        aligned_image = cv2.warpAffine(aligned_image, M, (aligned_image.shape[1], aligned_image.shape[0]))

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
    print("pre alignment image shape")
    print(img.shape)
    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()
    differenceImages[i] = align_images_multi_scale(firstImg, img)

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


