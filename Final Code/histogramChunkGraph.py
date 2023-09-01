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
plottingIntensities = []
standardDeviations = []
errors = []
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

    #try stretch cropped image's histogram
    p2, p98 = np.percentile(croppedImage, (0.5, 99.5))
    #print("p2 = " + str(p2))
    #print("p98 = " + str(p98))
    #croppedImage = croppedImage - p2
    croppedImage = np.clip(croppedImage, p2, p98)

    mean = np.nanmean(croppedImage)

    croppedImage = croppedImage - mean


    kernalSize = 5
    kernel = np.ones([kernalSize, kernalSize])
    kernel = kernel / ((kernalSize**2) - 3)
    croppedImage = ndimage.convolve(croppedImage, kernel)
    croppedImage = bilateral_filter(croppedImage, 5, 1, 15)
    
    #croppedImage = ndimage.median_filter(croppedImage, size=5, mode = "nearest")
    croppedImage = ndimage.convolve(croppedImage, kernel)

    mean = np.nanmean(croppedImage)

    croppedImage = croppedImage - mean

    #croppedImage = croppedImage - mean

    if i == 0:

        firstImg = croppedImage

        plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)

        #defines slider axis
        axs = plt.axes([0.15, 0.0001, 0.65, 0.03])
        sb = Slider(axs, 'max intensity', -0.03, 0.03, valinit=0, valstep = 0.000001)

        xPos = 0

        ax1.hist(firstImg)
        #runs when slider moved
        def update(val):
            global xPos
            ax1.clear()
            ax1.hist(firstImg)
            xPos = sb.val

            ax1.plot([xPos, xPos], [0, 500])
        
            plt.draw()

        ax1.set_title("differences")

        sb.on_changed(update)
        plt.show()
        maxX = xPos
        print("max intensity = " + str(maxX))

        plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)

        #defines slider axis
        axs = plt.axes([0.15, 0.001, 0.65, 0.03])
        sb = Slider(axs, 'min intensity', -0.03, 0.03, valinit=0, valstep = 0.0001)
        xPos = 0


        ax1.hist(firstImg)
        #runs when slider moved
        def update(val):
            global yPos
            ax1.clear()
            ax1.hist(firstImg)

            xPos = sb.val

            ax1.plot([xPos, xPos], [0, 500])
        
            plt.draw()


        sb.on_changed(update)
        plt.show()
        minX = xPos
        print("min intensity = " + str(minX))
        #minX = 0.01
        #maxX = 0.05
        minX = -0.004
        maxX = 0.004

        positions = np.argwhere(np.logical_and(croppedImage > minX, croppedImage < maxX))

        print("positions ")
        print(positions)
        print("end positions")

        #plt.figure()
        #plt.axis("off")

        #ax1 = plt.subplot(1, 2, 1)
        #ax1.imshow(firstImg, cmap="gray")
        #ax1.plot(positions[1][:], positions[0][:],  "*")


    intensities = []
    print("cropped image shape")
    print(croppedImage.shape)
    print("no of points = " + str(len(positions)))

    for point in positions:
        #print("point = " + str(point))
        #point = [positions[1][x], positions[0][x]]
        #print("x = " + str(point[0]-1))
        #print("y = " + str(point[1] - 1))
        
        intensity = croppedImage[point[0]-1, point[1]-1]
        intensities = intensities + [intensity]

    intensities = np.array(intensities)
    plottingIntensities = plottingIntensities + [np.nanmean(intensities)]

    currentError = np.nanvar(intensities) / len(intensities)   #output was tiny
    currentError = currentError ** 0.5


    #currentError = np.nanstd(intensities)

    standardDeviations = standardDeviations + [np.std(intensities)]


    errors = errors + [currentError]

    newImg = croppedImage

    img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = newImg

    differenceImages[i] = img

    #secondImgHist = np.histogram(croppedImage.ravel(), bins = 100, range=(-0.03, 0.02), density=True, weights=None)
    #secondImgHist = np.histogram(croppedImage.ravel(), bins = 100, density=True, weights=None)

#angles = np.array([50, 50, 80, 100, 110, 110, 140])
#angles = np.array(angles)
#angles = angles - 50

angles = np.array([0, 0, 30, 45, 60, 60, 90])

plt.plot(angles, plottingIntensities, "*")
plt.show()


plt.figure()
plt.axis("off")

offset = -50

ax1 = plt.subplot(1, 2, 1)

print("out of loop - should plot stuff now")      
#angles = np.array([50, 50, 80, 100, 110, 110, 140])
#angles = np.array(angles)
angles = np.array([0, 0, 30, 45, 60, 60, 90])

x_new = np.linspace(np.min(angles), np.max(angles), 100)
x_new_rad = np.deg2rad(x_new + offset)



def objective(x, a, b, c):
    x = np.deg2rad(x)
    return a * np.sin((2*x) + b) + c

p0 = 0.02, -0.5 , 0   #close to currve - should give better fit.

#popt, _ = curve_fit(objective, x_vals, y_vals)


popt, _ = curve_fit(objective, angles, plottingIntensities, p0, sigma=standardDeviations)

#up = c
amplitude, offset, up = popt

y_new = objective(x_new, amplitude, offset, up)
print("angles")
print(angles)


#parameters of fit curve
print("offset = " + str(offset))
print("amplitude = " + str(amplitude))
print("c = " + str(up))

ax1.plot(x_new, y_new)
ax1.errorbar(angles, plottingIntensities, yerr=errors, fmt="*")
ax1.plot(angles[0:len(plottingIntensities)], plottingIntensities, "*")


print("errors")
print(errors)
plt.tight_layout()

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

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(firstImg, cmap="gray")

for point in positions:
        ax1.plot(point[1]-1, point[0]-1, "*")

plt.show()


print("angles")
print(angles)

