import numpy as np
import h5py
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave


def getGradient (img):

    gradX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gradY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    grad = cv2.addWeighted(np.absolute(gradX), 0.5, np.absolute(gradY), 0.5, 0)
    return grad

#ECC algorithm matches based on patterns
#ran on raw image, then on image intensities to try and fix last two.

imageNumbers = np.arange(0, 7)

#used to store images
differenceImages = []
intensityImages = []
alignedImages = []
untouchedDifferences = []
testImages = []
untouchedTestImages = []

#reading images from file
for i in imageNumbers: 

    differenceImg = io.imread(r"finalCross/alignedCrossNewCentreTest28_D_%s.tif" % str(i))
    intensityImg = io.imread(r"finalCross/alignedCrossNewCentreTest28_I_%s.tif" % str(i))

    #store both, so not rescaled
    differenceImg = np.array(differenceImg)
    untouchedDifferences = untouchedDifferences + [differenceImg]

    #adjusting contrast of display image
    checkImg = np.clip(differenceImg, -0.08, 0.1)
    differenceImg = (checkImg-np.nanmin(checkImg))/(np.nanmax(checkImg)-np.nanmin(checkImg))
    differenceImg = differenceImg * 255

    #substitute for dead pixels
    intensityImg = np.nan_to_num(intensityImg)
    differenceImg = np.nan_to_num(differenceImg)

    #store images
    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

sz = differenceImages[0].shape


# Define the motion model


#options are:
#   cv2.MOTION_TRANSLATION - limit to x,y shift
#   cv2.MOTION_EUCLIDEAN - limit to rotation and x, y translation
#   cv2.MOTION_AFFINE - rotate, translate, scale, and shear
#   cv2.MOTION_HOMOGRAPHY - accounts for some 3d effects, as well as the above

warp_mode = cv2.MOTION_AFFINE

#empty matrix to contiain transformation
warp_matrix = np.eye(2, 3, dtype=np.float32)

im1 = np.array(differenceImages[0], dtype = np.float32)
alignedImages = alignedImages + [untouchedDifferences[0]]

for i in range(1, len(differenceImages)):

    im2 = np.array(differenceImages[i], dtype = np.float32)

    #if images weren't already greyscale, would need to convert them here
    im1_gray = im1
    im2_gray = im2
 
    # Specify the number of iterations.
    #more = more accurate but slower
    number_of_iterations = 5000
 
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-9
 
    # Define termination criteria 
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
    # Run the ECC algorithm. Transformation stored in wrap matrix
    #loops to find the best match

    #to use pixel vals
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    #use gradient instead
    #(cc, warp_matrix) = cv2.findTransformECC(grad1, grad2, warp_matrix, warp_mode, criteria)

    im2 = untouchedDifferences[i]
    
    #just needed in case of 3x3 matrix transform
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    alignedImages = alignedImages + [im2_aligned]

    #show transformaiton matrix used
    print("M = ")
    print(warp_matrix)

ax1 = plt.subplot(1, 2, 1)

img1 = alignedImages[0]

fig1 = ax1.imshow(img1, cmap="gray")


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)


#runs when slider moved
def update(val):
    img1 = alignedImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")

print("plotted")
sb.on_changed(update)

plt.show()

#saves all aligned images
for i in range(0, len(alignedImages)):
    sk_imsave(r"finalCross/testFirst_D_%s.tif" %str(i), alignedImages[i])
