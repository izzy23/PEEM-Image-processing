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
    image = np.clip(image, 120, 140)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255
    image = np.array(image, dtype="uint8")

    plt.imshow(image, cmap="gray")
    plt.show()

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
print("length before corrections")
print(len(intensityImages))


#differenceImages = np.array(differenceImages, dtype=np.float32)
#intensityImages = np.array(intensityImages, dtype=np.float32)

img1 = differenceImages[0]


correctedDifferenceImages = []
correctedIntensityImages = []
for i in range(1, len(differenceImages)):

    img2 = differenceImages[i]

    
    # Initiate SIFT detector
    sift = cv2.SIFT_create(sigma = 7, nOctaveLayers = 3)

    #orb = cv2.ORB_create(nfeatures=2000)
    #kp1, des1 = orb.detectAndCompute(img1, None)
    #kp2, des2 = orb.detectAndCompute(img2, None)

    #kp_img = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    #plt.imshow(kp_img, cmap="gray")
    #plt.show
    
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



    # FLANN parameters
    # I literally copy-pasted the defaults
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)   # or pass empty dictionary
    # do the matching
    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    #matches = flann.knnMatch(des1,des2,k=2)

    print("points available")
    print(len(des1))
    print(len(des2))
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    #matches = sorted(matches, key=lambda x: x.distance)

    print("displaying match image")
    #match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    


    # store only good matches as before in good
    #"ratio test"
    good = []
    for m, n in matches:
        #for m in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
        #good.append(m)
    points1 = []
    points2 = []
    tolerance = 400
    for match in good:

        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt

        if abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance:
            points1 = points1 + [p1]
            points2 = points2 + [p2]
    points1 = np.array(points1, dtype = np.float32)
    points2 = np.array(points2, dtype = np.float32)
   

    
    MIN_MATCH_COUNT = 4
    if len(points1)>MIN_MATCH_COUNT:
        # extract coordinated for query and train image points
        src_pts = points1
        dst_pts = points2
        #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        print("points")
        print(src_pts)
        # use homography to get the M transformation matrix
        #not working for rotations
        #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)  #returning 3d transformation matrix for some reason - crop out below to make work with rest of code
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
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
        print("showing moved image")
        plt.imshow(im2, cmap="gray")
        plt.show()
        print("new image!")


        correctedDifferenceImages = correctedDifferenceImages + [im2]
    else:
        print("bad match")
        correctedDifferenceImages = correctedDifferenceImages + [img2]

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
#print("weird shape")
#print(correctedDifferenceImages.shape)

img1 = correctedDifferenceImages[0]


fig1 = ax1.imshow(img1, cmap="gray")


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")


sb.on_changed(update)
plt.show()