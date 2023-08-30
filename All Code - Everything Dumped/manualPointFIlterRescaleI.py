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
from matplotlib.widgets import RadioButtons




startNo = 0
endNo = 6

imageNumbers = np.arange(startNo + 1, endNo + 1)


imageNumbers = np.arange(1, 6)

differenceImages = [io.imread(r"finalCross/alignedCrossNewFinalFinalFinal_D_0.tif")]
intensityImages = [io.imread(r"finalCross/alignedCrossNewFinalFinalFinal_I_0.tif")]

for i in imageNumbers: 

    differenceImg = io.imread(r"finalCross/alignedCrossNewFinalFinalFinal_D_%s.tif" % str(i))
    intensityImg = io.imread(r"finalCross/alignedCrossNewFinalFinalFinal_I_%s.tif" % str(i))
    print("shape")
    print(differenceImg.shape)
    print(intensityImg.shape)
    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]


initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []


image = np.clip(intensityImages[0], 10, 25)
image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
intensityImages[0] = image * 255

initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []
first = True


for image in intensityImages:
    if first == False:
        #plt.imshow(image, cmap="gray")
        #plt.show()
        image = np.clip(image, 10, 25)
        image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
        image = image * 255

    else:
        first = False
    print("looping")
    #avg = np.nanmean(image)
    #std = np.nanstd(image)

    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    #image = image * 255

    #newImage = initialAvg + ((image - avg) * (initialStd / std))


    newImage = np.clip(image, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack




#differenceImages = np.array(differenceImages, dtype=np.float32)
#intensityImages = np.array(intensityImages, dtype=np.float32)

img1 = intensityImages[0]



correctedDifferenceImages = []
correctedIntensityImages = []
correctedDifferenceImages = correctedDifferenceImages + [differenceImages[0]]
correctedIntensityImages = correctedIntensityImages + [intensityImages[i]]
for i in range(1, len(intensityImages)):
    

    img2 = intensityImages[i]
    print("standard deviation = " + str(np.nanstd(img2)))

    
    # Initiate SIFT detector - semes to detect the most equal points so far
    #sift = cv2.SIFT_create(sigma = 9)
    sift = cv2.SIFT_create(sigma = 7)
    
    # find the keypoints and descriptors with SIFT
    # Here img1 and img2 are grayscale images
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print("no of current points = " + str(len(des2)))

    #if cross check = True, then applies an alternative to the ratio test used below - apparently shouldn't use both
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    #bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 9)
    search_params = dict(checks = 500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
 

    # store only good matches as before in good
    #"ratio test"
    good = []
    for m, n in matches:
        #for m in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

        #good.append(m)
    print("no of good matches = " + str(len(good)))


    #plt.show()
    print("good")
    print(good)
    
    # use homography to get the M transformation matrix
    #not working for rotations
    #M, mask = cv2.findHomography(des1, des2, cv2.RANSAC,5.0)  #returning 3d transformation matrix for some reason - crop out below to make work with rest of code
    points1 = []
    points2 = []
    tolerance = 800
    for match in good:

        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt

        if abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance:
            points1 = points1 + [p1]
            points2 = points2 + [p2]
    points1 = np.array(points1, dtype = np.float32)
    points2 = np.array(points2, dtype = np.float32)
    #M = cv2.estimateAffinePartial2D(des1, des2)
    approvedP1 = []
    approvedP2 = []
    j = 0

    fig = plt.figure()
    plt.axis("off")

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")
    fig1 = ax1.plot(points1[0][0], points1[0][1], marker='v', color="red")
    fig2 = ax2.plot(points2[0][0], points2[0][1], marker='v', color="red")
    

    rax = plt.axes([0.47, 0.5, 0.1, 0.15])
    radioHandle = RadioButtons(rax, ("yes", "no"))
    #radioHandle.on_clicked(buttonFunc)
    plt.tight_layout()

    
    def buttonFunc(label):
        global j
        global approvedP1
        global approvedP2
        global fig1
        global fig2
        
        if label == "yes":
            approvedP1 = approvedP1 + [points1[j]]
            approvedP2 = approvedP2 + [points2[j]]

        j = j + 1
        print("loop button")

        if j < len(points1):
            ax1.clear()
            ax2.clear()
            ax1.imshow(img1, cmap="gray")
            ax2.imshow(img2, cmap="gray")
            ax1.plot(points1[j][0], points1[j][1], marker='v', color="red")
            ax2.plot(points2[j][0], points2[j][1], marker='v', color="red")
            plt.draw()
        else:
            plt.close()

    radioHandle.on_clicked(buttonFunc)

    
    plt.show()

            
    
    points1 = np.array(approvedP1, dtype = np.float32)
    points2 = np.array(approvedP2, dtype = np.float32)

 
    
    #M, _ = cv2.estimateAffinePartial2D(points1, points2)
    if len(points2) > 2:
        M, _ = cv2.estimateAffinePartial2D(points2, points1)
        #M = M[0]


        w,h = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # use this transform to shift train image to query image
        #dst = cv2.getPerspectiveTransform(pts,M)  

        img2 = np.array(img2, dtype = np.float32)
        M = np.array(M, dtype = np.float32)
        #im2 = cv2.warpPerspective(img2, M, (h, w))
        print("no of images")
        print(len(differenceImages))
        print(len(intensityImages))
        img2 = np.array(differenceImages[i], dtype = np.float32)
        im2 = cv2.warpAffine(img2, M, (w,h))
        print("showing moved image")
        plt.imshow(im2, cmap="gray")
        plt.show()
        print("new image!")


        correctedDifferenceImages = correctedDifferenceImages + [im2]

        img2 = np.array(intensityImages[i], dtype = np.float32)
        im2 = cv2.warpAffine(img2, M, (w,h))
        print("showing moved image")
        plt.imshow(im2, cmap="gray")
        plt.show()
        print("new image!")


        correctedIntensityImages = correctedIntensityImages + [im2]
    else:
        correctedDifferenceImages = correctedDifferenceImages + [differenceImages[i]]
        correctedIntensityImages = correctedIntensityImages + [intensityImages[i]]

    

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