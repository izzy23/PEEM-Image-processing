import numpy as np
import h5py
import cv2
#import time
import io
#import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave


#array for images with domains
imageNumbers = ["321124-321125", "321127-321136", "321227-321236", "321457-321466", "321527-321536", "321617-321626", "321627-321636"]
differenceImages = []
intensityImages = []

for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

initialIntensityImages = np.array(intensityImages, dtype = np.float32)
initialDifferenceImages = np.array(differenceImages, dtype = np.float32)

imageStack = []


initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in initialDifferenceImages:
    image = image.astype(np.float32)
    #image = np.array(image, dtype = np.float32)
    

    image = (image + 1)/2
    image = image*255


    avg = np.nanmean(image)
    std = np.nanstd(image)

    newImage = initialAvg + ((image - avg) * (initialStd / std))
    image = newImage


    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255


    image = np.array(image, dtype = "uint8")
    image = cv2.equalizeHist(image)


    imageStack = imageStack + [image]

correctedDiff = imageStack


initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []

for image in intensityImages:
    print("looping")
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    #image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))

    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack
correctedDifferenceImages = []
differenceImages = np.array(correctedDiff, dtype = "uint8")
img1 = differenceImages[0]

for i in range(1, len(differenceImages)):

    img2 = differenceImages[i]

    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    #surf = cv2.SURF_cre
    
    # find the keypoints and descriptors with SIFT
    # Here img1 and img2 are grayscale images
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    #des1 = cv2.goodFeaturesToTrack(img1,
    #                                 maxCorners=200,
    #                                 qualityLevel=0.05,
    #                                 minDistance=30,   
    #                                 blockSize=3)
    #des2 = cv2.goodFeaturesToTrack(img2,
    #                                 maxCorners=200,
    #                                 qualityLevel=0.05,
    #                                 minDistance=30,   
    #                                 blockSize=3)

    print("no of keypoints")
    print(len(des1))
    print(len(des2))

    # FLANN parameters
    # I literally copy-pasted the defaults
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=500)   # or pass empty dictionary
    # do the matching
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store only good matches as before in good
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 4
    if len(good)>MIN_MATCH_COUNT:
        # extract coordinated for query and train image points
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        print("points")
        print(src_pts)
        # use homography to get the M transformation matrix
        #not working for rotations
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)  #returning 3d transformation matrix for some reason - crop out below to make work with rest of code
        #M = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        #M = M[:-1, :]
        
        matchesMask = mask.ravel().tolist()
        # here im1 is the original RGB (or BGR because of cv2) image
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # use this transform to shift train image to query image
        #dst = cv2.getPerspectiveTransform(pts,M)  
        #print("dst")
        #print(dst)
        #print("shape")
        #print(dst.shape)
        #dst = np.array(dst, dtype = np.float32)
        img2 = np.array(img2, dtype = np.float32)
        im2 = cv2.warpPerspective(img2, M, (h, w))
        plt.imshow(im2, cmap="gray")
        plt.show()


    #correctedDifferenceImages = correctedDifferenceImages + [[np.array(dst, dtype="uint8")]]
    # for now I only displayed it using polylines, depending on what you need you can use these points to do something else
    # I overwrite the original RGB (or BGR) image with a red rectangle where the smaller image should be
    #im2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)

    #print("shape")
    #print(img2.shape)
    #img2 = np.array(img2, dtype = np.float32)
    #im2 = cv2.warpAffine(img2, M, (w,h))
    #print("shape")
    #print(im2.shape)
    #plt.imshow(im2, cmap="gray")
    #plt.show()
    #cv2.imwrite('image_overlap.png', im2) 
    correctedDifferenceImages = correctedDifferenceImages + [im2]

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
#print("weird shape")
#print(correctedDifferenceImages.shape)

img1 = correctedDifferenceImages[0]


fig1 = ax1.imshow(img1, cmap="gray")


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")


sb.on_changed(update)
plt.show()