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


def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    
    return rotatedImg


#startNo = 0
#endNo = 6

#imageNumbers = np.arange(startNo + 1, endNo + 1)

#differenceImages = [io.imread(r"rotatedCross\%s_D.tif" % str(startNo)).astype(np.float32)]
#intensityImages = [io.imread(r"rotatedCross\%s_I.tif" % str(startNo)).astype(np.float32)]

#for i in imageNumbers: 

#    differenceImg = io.imread(r"rotatedCross\%s_D.tif" % str(i))
#    intensityImg = io.imread(r"rotatedCross\%s_I.tif" % str(i))

#    differenceImages = differenceImages + [differenceImg]
#    intensityImages = intensityImages + [intensityImg]

imageNumbers = ["321124-321125", "321127-321136", "321227-321236", "321457-321466", "321527-321536", "321617-321626", "321627-321636"]
#imageNumbers = ["321124-321125", "321127-321136", "321227-321236"]

differenceImages = []
intensityImages = []



for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))
    #plt.imshow(differenceImg, cmap="gray", vmin = -0.08, vmax = 0.1)
    #plt.show()

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

    
offsetAngle = 50

angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + offsetAngle
#intensityImages[0] = rotateImg(intensityImages[0], offsetAngle)
#differenceImages[0] = rotateImg(differenceImages[0], offsetAngle)ack
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(0, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])


    

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:

    image = np.clip(image, -0.08, 0.09)
    #image = cv2.equalizeHist(image)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255
    image = np.array(image, dtype="uint8")



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

    
    # Initiate SIFT detector - semes to detect the most equal points so far
    #sift = cv2.SIFT_create(sigma = 7)
    sift = cv2.SIFT_create(sigma = 7)

    #surf = cv2.SIFT_create()

    #kp1, des1 = surf.detectAndCompute(img1, None)
    #kp2, des2 = surf.detectAndCompute(img2, None)


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
    #des1 = cv2.cornerHarris(img1, blockSize=2, ksize=3, k=0.04)
    #des2 = cv2.cornerHarris(img2, blockSize=2, ksize=3, k=0.04)
    #des1 = cv2.dilate(des1, None)
    #des2 = cv2.dilate(des2, None)

    #sift = cv2.SIFT_create()
    #kp1, des1 = sift.detectAndCompute(img1, None)
    #kp2, des2 = sift.detectAndCompute(img2, None)

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

    print(len(des2))
    #if cross check = True, then applies an alternative to the ratio test used below - apparently shouldn't use both
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    #bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    

    print("displaying match image")
    #match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    #plt.imshow(match_img, cmap="gray")
    #plt.show()

    print("no before ratio test")
    print(len(matches))
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
    i = 0

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
        global i
        global approvedP1
        global approvedP2
        global fig1
        global fig2
        
        if label == "yes":
            approvedP1 = approvedP1 + [points1[i]]
            approvedP2 = approvedP2 + [points2[i]]

        i = i + 1
        if i < len(points1):
            ax1.clear()
            ax2.clear()
            ax1.imshow(img1, cmap="gray")
            ax2.imshow(img2, cmap="gray")
            ax1.plot(points1[i][0], points1[i][1], marker='v', color="red")
            ax2.plot(points2[i][0], points2[i][1], marker='v', color="red")
            plt.draw()
        else:
            plt.close()

    radioHandle.on_clicked(buttonFunc)

    
    plt.show()

            
    
    points1 = np.array(approvedP1, dtype = np.float32)
    points2 = np.array(approvedP2, dtype = np.float32)

 
    
    #M, _ = cv2.estimateAffinePartial2D(points1, points2)
    #print("m = ")
    #print(M)
    #M = cv2.estimateRigidTransform(points1, points2)

    dx = np.arange(-200, 200, 1)
    dy = dx

    bestDx = 0
    bestDy = 0
    print(points1)

    minScore = 99999999999

    #for dxLoop in dx:
    #    resultX = (points1[:, 0] + dxLoop) / points2[:, 0]
    #    for dyLoop in dy:
    #        resultY = (points1[:, 1] + dyLoop) / points2[:, 1]
    #        diff = resultY - resultX
    #        meanDiff = abs(np.mean(diff))
    #        std = np.std(np.concatenate((resultX, resultY)))
    #        std = abs(np.mean(diff))
    #        scoreDiff = abs((np.nanmean(points2[:, 0] / (points1[:, 0] + bestDx))) - (np.nanmean(points2[:, 1] / (points1[:, 1] + bestDx))))
    #        score = meanDiff + std + scoreDiff

            #result = np.concatenate((resultX, resultY))
            #print(len(resultY))
            #print(len(result))
            #std = np.var(result, dtype = np.float64)
    #        if score < minScore:
    #            minScore = score
    #            bestDx = dxLoop
    #            bestDy = dyLoop
                #print("min std = " + str(minStd))
    #print("min std = " + str(minScore))
    minStd = 9999999999999

    for k in dx:

        result = (points1[:, 0] + k) / points2[:, 0]

        #print("check found x")
        #print(points1[:][0])
        #print(points2[:][0])
        std = np.std(result, dtype = np.float64)

        if std < minStd:
            minStd = std
            bestDx = k
    
    minStd = 999999999
    for k in dy:
        result = (points1[:, 1] + k) / points2[:, 1]
        std = abs(np.var(result, dtype = np.float64))

        if std < minStd:
            minStd = std
            bestDx = k
    
    sx = np.nanmean(points2[:, 0] / (points1[:, 0] + bestDx))
    sy = np.nanmean(points2[:, 1] / (points1[:, 1] + bestDx))

    print("scale factors")
    print(sx)
    print(sy)
 
    s = (sx + sy) / 2
    print(s)

    M = np.array([[s, 0, (s*bestDx)], [0, s, (s*bestDy)]])

    #print(M)



    w,h = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # use this transform to shift train image to query image
    #dst = cv2.getPerspectiveTransform(pts,M)  

    img2 = np.array(img2, dtype = np.float32)
    M = np.array(M, dtype = np.float32)
    #im2 = cv2.warpPerspective(img2, M, (h, w))
    M, _ = cv2.estimateAffinePartial2D(points2, points1)
    M[0][1] = 0
    M[1][0] = 0
    print("M =")
    print(M)
    im2 = cv2.warpAffine(img2, M, (w,h))
    print("showing moved image")
    plt.imshow(im2, cmap="gray")
    plt.show()
    print("new image!")


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
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferenceImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")


sb.on_changed(update)
plt.show()