import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib import colorbar
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave
from matplotlib.widgets import RadioButtons
import scipy.optimize as scipy
from skimage.exposure import match_histograms
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_adapthist
from skimage.filters import median
import skimage.filters
from scipy import ndimage
from matplotlib.colors import ListedColormap
from skimage import data
from skimage import color
from skimage import img_as_float, img_as_ubyte


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

def histMatch(source, template):
    oldShape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_vals, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_vals, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_vals)

    return interp_t_values[bin_idx].reshape(oldShape)

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d


imageNumbers = np.arange(1, 7)

#differenceImages = [io.imread(r"finalCross/alignedCrossNewCentreTest22_D_0.tif")]
differenceImages = [io.imread(r"AlignedCross/alignedCross_D_0.tif")]
intensityImages = [io.imread(r"AlignedCross/alignedCross_I_0.tif")]

for i in imageNumbers: 

    #differenceImg = io.imread(r"finalCross/alignedCrossNewCentreTest22_D_%s.tif" % str(i))
    intensityImg = io.imread(r"AlignedCross/alignedCross_I_%s.tif" % str(i))
    differenceImg = io.imread(r"AlignedCross/alignedCross_D_%s.tif" % str(i))

    print("shape")
    print(differenceImg.shape)
    print(intensityImg.shape)


    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

angles = np.array([50, 50, 80, 100, 110, 110, 140])
angles = np.array(angles)
angles = angles - 50


print("reduced angles = " + str(angles))

img = differenceImages[0]
    
#select centre reigon to normalise with
displayImg = img
displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

# Select ROI
w, h = displayImg.shape

# Naming a window
cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

#resizes window, not actual image
cv2.resizeWindow("select ROI", w , h)

r = cv2.selectROI("select ROI", displayImg)
newImages = []

for i in range(0, len(differenceImages)):
    img = differenceImages[i]

    
    croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]


    hist = np.histogram(croppedImage.ravel(), bins = 100, range=(-0.03, 0.02), density=True, weights=None)
    plt.hist(croppedImage.ravel(), bins = 100)
    plt.show()

    if i == 0:
        firstImg = croppedImage
        #firstImgHist = np.histogram(croppedImage.ravel(), bins = 100, range=(-0.03, 0.02), density=True, weights=None)
        firstImgHist = np.histogram(croppedImage.ravel(), bins = 100, range=None, density=True, weights=None)
    
    #newImg = cv2.medianBlur(croppedImage, 5)
    #newImg = match_histograms((croppedImage), (firstImg))
    #newImg = histMatch(croppedImage, firstImg)
    #newImg = croppedImage

    #reasonable results with 18

    p2, p98 = np.percentile(croppedImage, (6, 99.9))

    newImg = croppedImage - p2

    newImages = newImages + [newImg]

    img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = newImg

    differenceImages[i] = img
    #secondImgHist = np.histogram(croppedImage.ravel(), bins = 100, range=(-0.03, 0.02), density=True, weights=None)
    secondImgHist = np.histogram(croppedImage.ravel(), bins = 100, density=True, weights=None)

    #openCV histogram comparison of 2 images
    #diff = cv2.compareHist(tuple(firstImgHist), tuple(secondImgHist), cv2.HISTCMP_CHISQR)
    #diff = chi2_distance(firstImgHist, secondImgHist)
    #print("diff = " + str(diff))

    
plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)

img1 = differenceImages[0]

fig1 = ax1.imshow(img1, cmap="gray")

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


differenceImages = newImages
#colourmap stuff!!

#define different colourmaps using anchorpoints
#cdict = {'red':   [[0.0,  0.0, 0.0],
#                   [0.5,  1.0, 1.0],
#                   [1.0,  1.0, 1.0]],
#         'green': [[0.0,  0.0, 0.0],
#                   [0.25, 0.0, 0.0],
#                   [0.75, 1.0, 1.0],
#                   [1.0,  1.0, 1.0]],
#         'blue':  [[0.0,  0.0, 0.0],
#                   [0.5,  0.0, 0.0],
#                   [1.0,  1.0, 1.0]]}

#newcmp = LinearSegmentedColormap("testCmap", segmentdata=cdict, N = 256)

#these kinda work
#colourVal = 5
#redVals = np.append(np.linspace(colourVal, 0, 60), np.zeros(300))
#greenVals = np.append(np.zeros(30), np.linspace(0, colourVal, 60))
#greenVals = np.append(greenVals, np.linspace(colourVal, 0, 60))
#greenVals = np.append(greenVals, np.zeros(210))
#blueVals = np.append(np.zeros(60), np.linspace(0, colourVal, 60))
#blueVals = np.append(blueVals, np.linspace(30, 0, 60))
#blueVals = np.append(blueVals, np.zeros(180))



colourVal = 6
redVals = np.append(np.linspace(colourVal, 0, 40), np.zeros(320))
greenVals = np.append(np.zeros(20), np.linspace(0, colourVal, 40))
greenVals = np.append(greenVals, np.linspace(colourVal, 0, 40))
greenVals = np.append(greenVals, np.zeros(260))
blueVals = np.append(np.zeros(40), np.linspace(0, colourVal, 40))
blueVals = np.append(blueVals, np.linspace(30, 0, 40))
blueVals = np.append(blueVals, np.zeros(240))

colouredImages = []
for i in range(0, len(differenceImages)):
     image = color.gray2rgb(differenceImages[i])
     currentAngle = angles[i]
     r = redVals[int(currentAngle)]
     g = greenVals[int(currentAngle)]
     b = blueVals[int(currentAngle)]

     colourMagnitude = ((r**2) + (g**2) + (b**2))**0.5
     r = (r / colourMagnitude) * colourVal
     g = (g / colourMagnitude) * colourVal
     b = (b / colourMagnitude) * colourVal
     colouredImage = image * [r, g, b]
     colouredImages = colouredImages + [colouredImage]
colouredImages = np.array(colouredImages, dtype = np.float64)
differenceImages = colouredImages
   
plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)

img1 = differenceImages[0]

fig1 = ax1.imshow(img1)

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

average = differenceImages[0]
average = ndimage.median_filter(average, size=5)


#average = averageImages(differenceImages)
colours = []
for i in range(1, len(differenceImages)):
     img = differenceImages[i]
     average = average.astype(float) + img.astype(float)

plt.imshow(average, vmin = 0, vmax = 0.1)
plt.show()