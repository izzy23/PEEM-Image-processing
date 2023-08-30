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
imageNumbers = np.arange(1, 6)

differenceImages = [io.imread(r"finalCross/alignedCrossNewFinal5_D_0.tif")]
intensityImages = [io.imread(r"finalCross/alignedCrossNewFinal5_I_0.tif")]

for i in imageNumbers: 

    differenceImg = io.imread(r"finalCross/alignedCrossNewFinal5_D_%s.tif" % str(i))
    intensityImg = io.imread(r"finalCross/alignedCrossNewFinal5_I_%s.tif" % str(i))
    print("shape")
    print(differenceImg.shape)
    print(intensityImg.shape)
    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

#intensityImages[0] = rotateImg(intensityImages[0], offsetAngle)
#differenceImages[0] = rotateImg(differenceImages[0], offsetAngle)ack
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)



    

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:

    image = np.clip(image, -0.08, 0.1)
    #image = cv2.equalizeHist(image)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255
    image = np.array(image, dtype="uint8")



    imageStack = imageStack + [image]
#imageStack = standardizeStack(imageStack)

differenceImages = imageStack
print("no of images")
print(len(differenceImages))

image = np.clip(intensityImages[0], 10, 30)
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
        image = np.clip(image, 10, 30)
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
print("length before corrections")
print(len(intensityImages))


#differenceImages = np.array(differenceImages, dtype=np.float32)
#intensityImages = np.array(intensityImages, dtype=np.float32)

img1 = intensityImages[0]
plt.imshow(img1, cmap="gray")
plt.show()
w, h = img1.shape
firstImg = [[np.zeros(w), np.zeros(h)], [np.zeros(w), np.zeros(h)]]
for i in range(0, 1):
     # Naming a window
    cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

    #resizes window, not actual image
    cv2.resizeWindow("select ROI", w , h)

    r1 = cv2.selectROI("select ROI", img1)
    r1 = np.array(r1)

    firstImg[i] = img1[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]
    



correctedDifferenceImages = []
correctedIntensityImages = []
for i in range(1, len(intensityImages)):
    img2 = intensityImages[i]
    print("standard deviation = " + str(np.nanstd(img2)))
    plt.imshow(img2, cmap="gray")
    plt.show()

    
    # Initiate SIFT detector - semes to detect the most equal points so far
    #sift = cv2.SIFT_create(sigma = 7)
    sift = cv2.SIFT_create(sigma = 7)

    w, h = img1.shape
    secondImg = [[np.zeros(w), np.zeros(h)], [np.zeros(w), np.zeros(h)]]

    for i in range(0, 1):
        w, h = img2.shape

        #img = differenceImages[i]
        # Naming a window
        cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

        #resizes window, not actual image
        cv2.resizeWindow("select ROI", w , h)

        r2 = cv2.selectROI("select ROI", img2)
        r2 = np.array(r2)

        secondImg = img2[int(r2[1]):int(r2[1]+r2[3]), int(r2[0]):int(r2[0]+r2[2])]
        secondImg = np.array(secondImg, dtype = "uint8")
    
        # find the keypoints and descriptors with SIFT
        # Here img1 and img2 are grayscale images
        #kp1, des1 = sift.detectAndCompute(img1,None)
        #kp2, des2 = sift.detectAndCompute(img2,None)
        kp1, des1 = sift.detectAndCompute(firstImg[0],None)
        kp2, des2 = sift.detectAndCompute(secondImg,None)

        print(len(des2))
        #if cross check = True, then applies an alternative to the ratio test used below - apparently shouldn't use both
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        #bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 7)
        search_params = dict(checks = 500)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)


        print("no before ratio test")
        print(len(matches))
        # store only good matches as before in good
        #"ratio test"
        good = []
        for m, n in matches:
            #for m in matches:
            if m.distance < 0.95*n.distance:
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
        k = 0

        fig = plt.figure()
        plt.axis("off")

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        ax1.imshow(firstImg[i], cmap="gray")
        ax2.imshow(secondImg, cmap="gray")
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
                ax1.imshow(firstImg, cmap="gray")
                ax2.imshow(secondImg, cmap="gray")
                ax1.plot(points1[i][0], points1[i][1], marker='v', color="red")
                ax2.plot(points2[i][0], points2[i][1], marker='v', color="red")
                plt.draw()
            else:
                plt.close()

        radioHandle.on_clicked(buttonFunc)

    
        plt.show()

            
    
        points1 = np.array(approvedP1, dtype = np.float32)
        points2 = np.array(approvedP2, dtype = np.float32)

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