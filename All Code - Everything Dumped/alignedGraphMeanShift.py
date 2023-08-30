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
from skimage.exposure import match_histograms
from skimage.exposure import rescale_intensity
from skimage import exposure
import skimage.filters
from scipy import ndimage
from scipy.optimize import curve_fit
import math


 
def vec_gaussian(img: np.ndarray, variance: float) -> np.ndarray:
    # For applying gaussian function for each element in matrix.
    sigma = math.sqrt(variance)
    cons = 1 / (sigma * math.sqrt(2 * math.pi))
    return cons * np.exp(-((img / sigma) ** 2) * 0.5)
 
 
def get_slice(img: np.ndarray, x: int, y: int, kernel_size: int) -> np.ndarray:
    half = kernel_size // 2
    return img[x - half : x + half + 1, y - half : y + half + 1]
 
 
def get_gauss_kernel(kernel_size: int, spatial_variance: float) -> np.ndarray:
    # Creates a gaussian kernel of given dimension.
    arr = np.zeros((kernel_size, kernel_size))
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            arr[i, j] = math.sqrt(
                abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2
            )
    return vec_gaussian(arr, spatial_variance)
 
 
def bilateral_filter(
    img: np.ndarray,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.ndarray:
    img2 = np.zeros(img.shape)
    gaussKer = get_gauss_kernel(kernel_size, spatial_variance)
    sizeX, sizeY = img.shape
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):
 
            imgS = get_slice(img, i, j, kernel_size)
            imgI = imgS - imgS[kernel_size // 2, kernel_size // 2]
            imgIG = vec_gaussian(imgI, intensity_variance)
            weights = np.multiply(gaussKer, imgIG)
            vals = np.multiply(imgS, weights)
            val = np.sum(vals) / np.sum(weights)
            img2[i, j] = val
    return img2


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

for i in range(0, len(differenceImages)):
    img = differenceImages[i]
    
    croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    #plt.hist(croppedImage.ravel(), bins = 500, range=(-0.03, 0.03), fc="k", ec="k")
    #plt.show()


    #mean = np.nanmean(croppedImage)
    #std = np.nanstd(croppedImage)
    #img = (img - mean) / std

    #if i == 0:
    #    firstImage = img
    #else:
    #    img = match_histograms(img, firstImage)

    #img = cv2.equalizeHist(img)
    #croppedImage = exposure.equalize_adapthist(croppedImage)

    #try stretch cropped image's histogram
    p2, p98 = np.percentile(croppedImage, (0.5, 99.5))
    #print("p2 = " + str(p2))
    #print("p98 = " + str(p98))
    #croppedImage = croppedImage - p2
    croppedImage = np.clip(croppedImage, p2, p98)

    mean = np.nanmean(croppedImage)

    croppedImage = croppedImage - mean

    # A very simple and very narrow highpass filter
    #kernel = np.array([[-1, -1, -1, -1, -1],
    #               [-1,  1,  2,  1, -1],
    #               [-1,  2,  4,  2, -1],
    #               [-1,  1,  2,  1, -1],
    #               [-1, -1, -1, -1, -1]])


    #works for light domains
    #low pass filter
    #kernalSize = 5
    #kernel = np.ones([kernalSize, kernalSize])
    #kernel = kernel / (kernalSize**2)
    #croppedImage = ndimage.convolve(croppedImage, kernel)

    
    #croppedImage = ndimage.median_filter(croppedImage, size=5, mode = "nearest")
    #croppedImage = ndimage.convolve(croppedImage, kernel)

    #end of section

    kernalSize = 5
    kernel = np.ones([kernalSize, kernalSize])
    kernel = kernel / ((kernalSize**2) - 5)
    croppedImage = ndimage.convolve(croppedImage, kernel)
    


    #size = 5 - good results from median
    #works with spatial variance = 1
    croppedImage = ndimage.median_filter(croppedImage, size=5, mode = "nearest")
    croppedImage = bilateral_filter(croppedImage, 19, 5, 31)
    croppedImage = ndimage.convolve(croppedImage, kernel)

    mean = np.nanmean(croppedImage)

    croppedImage = croppedImage - mean


    #croppedImage = ndimage.gaussian_filter(croppedImage, sigma=3, mode="nearest")
    #croppedImage = ndimage.fourier_gaussian(croppedImage, sigma=0.5)
    #croppedImage = ndimage.median_filter(croppedImage, size=3, mode = "nearest")
    #croppedImage = rescale_intensity(croppedImage, in_range=(p2, p98), out_range=(-0.04, 0.04 ))
    #mean = np.nanmean(croppedImage)

    #croppedImage = croppedImage - mean

    if i == 0:

        firstImg = croppedImage

    newImg = croppedImage
    #croppedImage = match_histograms(croppedImage, firstImg)
    plt.hist(croppedImage.ravel(), bins = 1000)
    plt.show()

    img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = newImg

    differenceImages[i] = img

    
    #secondImgHist = np.histogram(croppedImage.ravel(), bins = 100, range=(-0.03, 0.02), density=True, weights=None)
    secondImgHist = np.histogram(croppedImage.ravel(), bins = 100, density=True, weights=None)


    
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

rVals = []
noPoints = 3
for i in range(0, noPoints):
    #displayImg = np.clip(differenceImages[0], -0.08, 0.1)
    displayImg = differenceImages[0]
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

    # Select ROI
    w, h = displayImg.shape

    # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    r = cv2.selectROI("select ROI", displayImg)
    rVals = rVals + [r]


plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)

offset = 20
#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', -90, 90, valinit=0, valstep = 1)

def plotting(offset):

    global differenceImages
    global sb
    global axs
    global reigonIntensities
    global x_vals
    global y_vals

    angles = np.array([50, 50, 80, 100, 110, 110, 140])
    angles = np.array(angles)

    x_vals = []
    y_vals = []


    for x in range(0, 3):
        reigonIntensities = []

        r = rVals[x]
        print("current r = " + str(r))


        i = 0
        for box in differenceImages:
            # Crop image to selected reigon
            croppedImage = box[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            intensity = np.average(croppedImage)
            reigonIntensities = reigonIntensities + [intensity]
            y_vals = y_vals + [intensity]
            x_vals = x_vals + [angles[i]]
            i = i + 1

        #ax1.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
        ax1.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")
        #ax1.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
        
        
        ax1.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")
        
angles = np.array([50, 50, 80, 100, 110, 110, 140])
angles = np.array(angles)
x_new = np.linspace(np.min(angles), np.max(angles), 100)
x_new_rad = np.deg2rad(x_new + offset)

#amplitude1 = np.nanmax(reigonIntensities) 
#amplitude2 = abs(np.nanmin(reigonIntensities))
#amplitude = np.max([amplitude1, amplitude2]) * 0.8
#y_new = amplitude * np.sin(2 * x_new_rad)
plotting(offset)
#print("offset = " + str(offset))
def objective(x, a, b):
    x = np.deg2rad(x)
    return a * np.sin((2*x) + b)

popt, _ = curve_fit(objective, x_vals, y_vals)

amplitude, offset = popt

y_new = objective(x_new, amplitude, offset)

print("offset = " + str(offset))
print("amplitude = " + str(amplitude))

ax1.plot(x_new, y_new)
ax1.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")




plt.tight_layout()

sb.on_changed(update)
plt.show()


