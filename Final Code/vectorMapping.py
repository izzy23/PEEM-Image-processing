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


imageNumbers = np.arange(1, 7)

#reading in final aligned images
differenceImages = [io.imread(r"AlignedCross/alignedCross_D_0.tif")]
intensityImages = [io.imread(r"AlignedCross/alignedCross_I_0.tif")]
positions = []

for i in imageNumbers: 

    #differenceImg = io.imread(r"finalCross/alignedCrossNewCentreTest22_D_%s.tif" % str(i))
    intensityImg = io.imread(r"AlignedCross/alignedCross_I_%s.tif" % str(i))
    differenceImg = io.imread(r"AlignedCross/alignedCross_D_%s.tif" % str(i))

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]
    
    img = differenceImages[0]
    
#select centre reigon to normalise intensity with
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

    #finds top and bottom 0.5% of intensity histogram
    p2, p98 = np.percentile(croppedImage, (0.5, 99.5))

    #removes outlying intensities
    croppedImage = np.clip(croppedImage, p2, p98)

    #image's average intensity value
    mean = np.nanmean(croppedImage)

    #shifts average intensity value to 0
    croppedImage = croppedImage - mean

    #low pass filter
    kernalSize = 5
    kernel = np.ones([kernalSize, kernalSize])
    kernel = kernel / ((kernalSize**2) - 3)
    croppedImage = ndimage.convolve(croppedImage, kernel)

    #smoothes reigons contained by corners - corners are left untouched
    croppedImage = bilateral_filter(croppedImage, 5, 1, 15)
    
    #croppedImage = ndimage.median_filter(croppedImage, size=5, mode = "nearest")

    #reapplies low pass filter
    croppedImage = ndimage.convolve(croppedImage, kernel)

    #ensures mean intensity is still 0
    mean = np.nanmean(croppedImage)
    croppedImage = croppedImage - mean


    if i == 0:

        firstImg = croppedImage
        xPos = 0
        
        maxX = xPos
        print("max intensity = " + str(maxX))

        minX = xPos
        print("min intensity = " + str(minX))

        imgX, imgY = croppedImage.shape

positions = np.argwhere(np.logical_and(croppedImage > -1, croppedImage < 1))

intensities = []
print("cropped image shape")
print(croppedImage.shape)
print("no of points = " + str(len(positions)))

angles = np.array([0, 0, 30, 45, 60, 60, 90])   #angles from spreadsheet

#loops through all positiosn in cropped image (centre of cross)
for point in positions:
    #array of intensity values for current pixel
    intensities = []

    #loops through each image (each rotation)
    for j in range(0, len(differenceImages)):
        img = differenceImages[j]
        croppedImage = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        #pixel intensity at curretn rotation
        intensity = croppedImage[point[0]-1, point[1]-1]
        intensities = intensities + [intensity]

    intensities = np.array(intensities)
    
    plottingIntensities = intensities

    #currentError = np.nanstd(intensities)

    x_new = np.linspace(np.min(angles), np.max(angles), 100)    #curve fitted to 100 x values in angle range (0 -> 90)
    x_new_rad = np.deg2rad(x_new)   #numpy sin(x) needs radians

    #curve fitting
    def objective(x, a, b, c):
        x = np.deg2rad(x)
        return a * np.sin((2*x) + b) + c

    p0 = 0.02, -0.5 , 0   #close to currve - gives slightly better fit

    #min and max values allowed for each parameter (a, b, c) - ensures phase found correctly by forcing positive amplitude
    low = [0, -100, -100]
    high = [100, 100, 100]

    #finds curve fit
    popt, _ = curve_fit(objective, angles, plottingIntensities, p0, bounds=(low, high))

    #y shift = c = up
    amplitude, offset, up = popt
    
    #replaces pixel value in cropped image with phase found
    croppedImage[point[0]-1, point[1]-1] = offset

    #to update new image - used for vector map
    newImg = croppedImage

    img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = newImg

    differenceImages[0] = img


print("showing offsets")

domainStuff = differenceImages[0]
domainStuff = domainStuff[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
plt.imshow(domainStuff)
plt.show()

#save image
sk_imsave("offsetMappedPositive.tif", domainStuff)  #This is the vector mapping image - contains phase shift for each pixel in selected reigon!!

angles = np.array([0, 0, 30, 45, 60, 60, 90])

plt.plot(angles, plottingIntensities, "*")  #displays graph for final pixel - don't need
plt.show()

