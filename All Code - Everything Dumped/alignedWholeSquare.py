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
from skimage.exposure import equalize_adapthist
from skimage.filters import median
import skimage.filters
from scipy import ndimage


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

means = []
angles = np.array([50, 50, 80, 100, 110, 110, 140])
angles = np.array(angles)

#select centre reigon to normalise with
displayImg = differenceImages[0]
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

    meanVal = np.nanmean(croppedImage)

    means = means + [meanVal]

plt.plot(angles, means, "*")
plt.show()

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
for i in range(0, 3):
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

    angles = np.array([50, 50, 80, 100, 110, 110, 140])
    angles = np.array(angles)


    for x in range(0, 1):
        reigonIntensities = []

        r = rVals[x]
        print("current r = " + str(r))


        for box in differenceImages:
            # Crop image to selected reigon
            croppedImage = box[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            intensity = np.average(croppedImage)
            reigonIntensities = reigonIntensities + [intensity]

        reigonIntensities = np.array(reigonIntensities)
        #ax1.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
        ax1.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")
        #ax1.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
        x_new = np.linspace(np.min(angles), np.max(angles), 100)
        x_new_rad = np.deg2rad(x_new + offset)
        amplitude = np.nanmax(((reigonIntensities)**2)**0.5) * 0.7
        y_new = amplitude * np.sin(2 * x_new_rad)

        ax1.plot(x_new, y_new)
        ax1.plot(angles[0:len(reigonIntensities)], reigonIntensities, "*")
        

plotting(offset)

#runs when slider moved
def update(val):
    ax1.clear()
    offset = sb.val
    plotting(offset)
        
    plt.draw()


plt.tight_layout()

sb.on_changed(update)
plt.show()
