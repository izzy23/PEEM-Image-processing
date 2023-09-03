import numpy as np
import io
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
import skimage
from skimage import io
import matplotlib.colors
from scipy import ndimage
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

#vals  = [0, 0.327, 2.54, 2.841, 2.904, 3.21]
cvals  = [0, 0.327, 1, 2.77, 2.84, 3.21]

colors = ["#4285f4", "#4285f4", "#FBBC05", "#FBBC05", "#EA4335", "#EA4335"]

norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

image = io.imread(r"offsetMappedPositive.tif").astype(np.float32)

image = image / 2
image = np.array(image)

image = ndimage.median_filter(image, size=15)
#image = bilateral_filter(image, 3, 1, 3)
#image = bilateral_filter(image, 15, 1, 15)
#image = ndimage.median_filter(image, size=7)
plt.figure()

plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

ax = ax1.imshow(image, cmap=cmap, vmin = -1.530, vmax = 1.68)
plt.colorbar(ax)

ax2.hist(image)
plt.show()

