import numpy as np
import h5py
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave
from matplotlib.widgets import RadioButtons
from skimage import measure

from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


#reads images in stack
startNo = 0
endNo = 6

imageNumbers = np.arange(startNo + 1, endNo + 1)

differenceImages = [io.imread(r"rotatedCross\%s_D.tif" % str(startNo)).astype(np.float32)]
intensityImages = [io.imread(r"rotatedCross\%s_I.tif" % str(startNo)).astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"rotatedCross\%s_D.tif" % str(i))
    intensityImg = io.imread(r"rotatedCross\%s_I.tif" % str(i))

    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

    
#to apply transform to later - don't want to rescale intensities of these
initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

#used to standardise later intensities to
initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    #need NaN bc of dead pixels
    avg = np.nanmean(image)
    std = np.nanstd(image)
    
    #standardise intensity of current image
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255

    #atatype needed for openCV
    image = np.array(image, dtype="uint8")

    #median blur removes some dead pixels, but adds slight stripe pattern to images
    #image = cv2.medianBlur(image, 5)

    #good contrast range
    image = np.clip(image, 110, 150)

    #can use this instead, but almost binarises image - adjusts contrast too drastically
    #image = cv2.equalizeHist(image)

    #applies contrast change
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255

    image = np.array(image, dtype="uint8")

    imageStack = imageStack + [image]
#imageStack = standardizeStack(imageStack)

differenceImages = imageStack

#repeat image standardisation on intensity images
initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []

for image in intensityImages:

    avg = np.nanmean(image)
    std = np.nanstd(image)

    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises between 0 and 1
    image = image * 255

    #force same mean and standard deviatio as initial image
    newImage = initialAvg + ((image - avg) * (initialStd / std))

    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8") #openCV datatype

    #remove some defects
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")
        
    imageStack = imageStack + [image]
 
intensityImages = imageStack


#can apply gaussian blur.  Didnt fix much
#imageD = ndi.gaussian_filter(differenceImages[0], 1)
#imageI = ndi.gaussian_filter(intensityImages[0], 1)


imageD = differenceImages[0]

#invert image as contour function perfers dark background
imageD = imageD * 255
imageD = 255 - imageD

#adjust contrast
imageD = np.clip(imageD, 90, 180)
imageD = (imageD-np.nanmin(imageD))/(np.nanmax(imageD)-np.nanmin(imageD)) #renormalises - after contrast adjust so shows
imageD = imageD * 255

#attempt at further reducing dead pixels and random noise in image
sigma_est = estimate_sigma(imageD, average_sigmas=True)
imageD = denoise_wavelet(imageD, rescale_sigma=True)                         

#uses 1st order gradient to produce image of edges
edges1 = feature.canny(imageD, sigma=7)

#array of positions of edges found in iamges
#fully_connected = high to try and eep contours connected
contours1 = measure.find_contours(edges1, fully_connected="high")

for i in range(1, len(differenceImages)):
    img = differenceImages[i]

    #invert image
    img = img * 255
    img = 255 - img

    #adjust contrast
    img = np.clip(img, 90, 180)
    img = (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img)) #renormalises - after contrast adjust so shows
    img = img * 255

    #attempt at further reducing dead pixels and random noise in image
    sigma_est = estimate_sigma(img, average_sigmas=True)
    img = denoise_wavelet(img, rescale_sigma=True)  

    #zooms checked
    scales = np.linspace(0.8, 1.3, 50)
    maxMatches = 0

    for scale in scales:

        cy, cx = img.shape
        cy = cy / 2
        cx = cx / 2

        #gets affine transform for zoom - doesn#t rotate
        rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, scale)

        #applies transform
        zoomedImg = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)                  

        #edge img of current img at current zoom
        edges2 = feature.canny(zoomedImg, sigma=7)

        #array of edge positions
        contours2 = measure.find_contours(edges2, fully_connected="high")

        #looks for matching edge positions between two images
        currentMatches = 0
        for contour in contours1:
            for contour2 in contours2:
                if contour.all() == contour2.all():
                    currentMatches = currentMatches + 1
        if currentMatches > maxMatches:
            bestZoom = scale

    #rescale current image
    cy, cx = [ i/2 for i in img.shape[:-1] ]

    #gets affine transform
    rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, scale)

    #apply transform
    differenceImages[i] = cv2.warpAffine(differenceImages[i], rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    plt.imshow(differenceImages[i], cmap="gray")
    plt.show()

fig, ax = plt.subplots()
ax.imshow(imageD, cmap= "gray")

#shows contour positions found in first image
for contour in contours1:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
print("no of contours = " + str(len(contours1)))

plt.show()






