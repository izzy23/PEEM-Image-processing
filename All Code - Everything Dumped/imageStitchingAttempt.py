from matplotlib import pyplot as plt
import numpy as np
from skimage import data, util, transform, feature, measure, filters, metrics
from skimage import io
from skimage.io import imsave as sk_imsave
import cv2


def match_locations(img0, img1, coords0, coords1, radius=5, sigma=8):
    """Match image locations using SSD minimization.

    Areas from `img0` are matched with areas from `img1`. These areas
    are defined as patches located around pixels with Gaussian
    weights.

    Parameters:
    -----------
    img0, img1 : 2D array
        Input images.
    coords0 : (2, m) array_like
        Centers of the reference patches in `img0`.
    coords1 : (2, n) array_like
        Centers of the candidate patches in `img1`.
    radius : int
        Radius of the considered patches.
    sigma : float
        Standard deviation of the Gaussian kernel centered over the patches.

    Returns:
    --------
    match_coords: (2, m) array
        The points in `coords1` that are the closest corresponding matches to
        those in `coords0` as determined by the (Gaussian weighted) sum of
        squared differences between patches surrounding each point.
    """
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    weights = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    weights /= 2 * np.pi * sigma * sigma

    match_list = []
    for r0, c0 in coords0:
        roi0 = img0[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1]
        roi1_list = [img1[r1 - radius:r1 + radius + 1,
                          c1 - radius:c1 + radius + 1] for r1, c1 in coords1]
        # sum of squared differences
        ssd_list = [np.sum(weights * (roi0 - roi1) ** 2) for roi1 in roi1_list]
        match_list.append(coords1[np.argmin(ssd_list)])

    return np.array(match_list)




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

    

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    avg = np.nanmean(image)
    std = np.nanstd(image)

    
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    #image = cv2.medianBlur(image, 5)
    #image = np.clip(image, 110, 150)
    #image = cv2.equalizeHist(image)
    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    #image = image * 255
    #image = np.array(image, dtype="uint8")



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

img_list = differenceImages








min_dist = 5
corner_list = [feature.corner_peaks(
    feature.corner_harris(img, sigma=7), threshold_rel=0.001, min_distance=min_dist, )
               for img in img_list]



img0 = img_list[0]
coords0 = corner_list[0]
matching_corners = [match_locations(img0, img1, coords0, coords1, min_dist)
                    for img1, coords1 in zip(img_list, corner_list)]

for dst in matching_corners:
    print("printing dst")
    print(dst)


src = np.array(coords0)
trfm_list = []
for dst in matching_corners:
    trfm_list = trfm_list + [measure.ransac((dst, src),
                            transform.EuclideanTransform, min_samples=3,
                            residual_threshold=50, max_trials=500)[0]]
print("trfm_list")
print(trfm_list)

fig, ax_list = plt.subplots(6, 2, figsize=(6, 9), sharex=True, sharey=True)
for idx, (im, trfm, (ax0, ax1)) in enumerate(zip(img_list, trfm_list,
                                                 ax_list)):
    ax0.imshow(im, cmap="gray")
    ax1.imshow(transform.warp(im, trfm), cmap="gray", vmin = 0.4, vmax = 0.55)

    if idx == 0:
        ax0.set_title("initial")
        #ax0.set_ylabel(f"Reference Image\n(PSNR={psnr_ref:.2f})")
        ax1.set_title("corrected")

    ax0.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax1.set_axis_off()

fig.tight_layout()
plt.show()