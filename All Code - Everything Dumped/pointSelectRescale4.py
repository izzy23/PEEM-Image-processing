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
from mpl_point_clicker import clicker
from matplotlib.widgets import RadioButtons
from math import sqrt


def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

tolerance = 70
min_angle = np.pi*0.22

def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) -
            (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d


def rdp(points, epsilon):

    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results


def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    
    return rotatedImg

def contourPrep(r, img, lowBinaryThreshold, highBinaryThreshold):
   

    currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    currentBox = (currentBox-np.nanmin(currentBox))/(np.nanmax(currentBox)-np.nanmin(currentBox))   #rescales intensity for box - don't remove!!
    currentBox = currentBox * 255
    currentBox = np.array(currentBox, dtype="uint8")
    scaledBox = currentBox
    currentBox = cv2.medianBlur(currentBox, 3)
    ret,thresh = cv2.threshold(currentBox,lowBinaryThreshold,highBinaryThreshold,0) #120, 255, 0
    currentBox = thresh

    edgeImg = cv2.Canny(currentBox, 0, 200)

    contours, higherarchy = cv2.findContours(currentBox, 1, cv2.CHAIN_APPROX_TC89_KCOS)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #print("sorted contours = " )
    #print(sorted_contours)

    return currentBox, scaledBox, edgeImg, sorted_contours

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

#angles = [0, 0, -30, -45, -50, -50, -90]
angles = [0, 0, -28, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + offsetAngle
#intensityImages[0] = rotateImg(intensityImages[0], offsetAngle)
#differenceImages[0] = rotateImg(differenceImages[0], offsetAngle)ack
#differenceImages[0] = rotateImg(initialDifferenceImages[0], 45)

for i in range(0, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])
    plt.imshow(intensityImages[i], cmap="gray", vmin = 5, vmax = 30)
    plt.show()

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)


img1 = differenceImages[0]
img2 = intensityImages[0]

fig1 = ax1.imshow(img1, cmap="gray")
fig2 = ax2.imshow(img2, cmap="gray", vmin = 10, vmax = 30)

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]
    img2 = intensityImages[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("differences")
ax2.set_title("intensities")

sb.on_changed(update)
plt.show()


initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

#select corners for zoom in
firstImgPositions = []
correctedDifferenceImages = []
correctedIntensityImages = []

correctedDifferenceImages = correctedDifferenceImages + [differenceImages[0]]
correctedIntensityImages = correctedIntensityImages + [intensityImages[0]]

for i in range(0, len(differenceImages)):
    
    newImgPositions = []
    displayImg = np.clip(intensityImages[i], 10, 30)
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))
    #displayImg = intensityImages[i]
    plt.imshow(displayImg, cmap="gray")
    plt.show()
    # Select ROI
    w, h = displayImg.shape
    #loop 4 times to select each corner
    for x in range(0, 4):
        lowBinaryThreshold = 120
        highBinaryThreshold = 255
         #img = differenceImages[i]
        # Naming a window
        cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

        #resizes window, not actual image
        cv2.resizeWindow("select ROI", w , h)

        r = cv2.selectROI("select ROI", displayImg)
        r = np.array(r)

        currentBox, scaledBox, edgeImg, sorted_contours = contourPrep(r, differenceImages[i], 120, 255)
        

        #while len(sorted_contours) < 1:
        #img = differenceImages[i]
        # Naming a window
        #cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

        #resizes window, not actual image
        #cv2.resizeWindow("select ROI", w , h)

        #r = cv2.selectROI("select ROI", displayImg)
        #r = np.array(r)
        #currentBox, scaledBox, edgeImg, sorted_contours = contourPrep(r, differenceImages[i], 120, 255)
        
        
        

        plt.figure()
        plt.axis("off")
        cv2.drawContours(scaledBox, sorted_contours, 0, (0,255,0), 1)

        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        #edgeImg[dst>0.2*dst.max()]=[255]
        ax2.imshow(currentBox, cmap = "gray")
        ax3.imshow(255 - scaledBox, cmap="gray")
        ax1.imshow(edgeImg)

        rax = plt.axes([0.47, 0.5, 0.1, 0.15])
        radioHandle = RadioButtons(rax, ("lower threshold", "higher threshold"))

        def buttonFunc(label):
            global lowBinaryThreshold
            global highBinaryThreshold

            global r
            global currentBox
            global scaledBox
            global edgeImg
            global sorted_contours
            

            if label == "lower threshold":
                lowBinaryThreshold = lowBinaryThreshold - 5
            if label == "higher threshold":
                lowBinaryThreshold = lowBinaryThreshold + 5

            currentBox, scaledBox, edgeImg, sorted_contours = contourPrep(r, differenceImages[i], lowBinaryThreshold, highBinaryThreshold)
            print("low binary threshold = " + str(lowBinaryThreshold))
            #corner finding stuff
            

           
            cv2.drawContours(scaledBox, sorted_contours, 0, (0,255,0), 1)
            
            ax1.clear()
            ax2.clear()
            ax3.clear()

            ax2.imshow(currentBox, cmap = "gray")
            ax3.imshow(255 - scaledBox, cmap="gray")
            ax1.imshow(edgeImg)
            plt.draw()

        radioHandle.on_clicked(buttonFunc)

        plt.show()

        #found contours, next need to place corner in image
        print("i = " + str(i))
        print("x = " + str(x))

        if i == 0:
            img1 = scaledBox

            fig, ax = plt.subplots(constrained_layout=True)
            ax.imshow(img1, cmap="gray")
            klicker = clicker(ax, ["event"], markers=["x"])
            plt.show()

            positions1 = klicker.get_positions()
            positions1 = np.array(positions1["event"], dtype = np.float32)
            print("selected position = " )
            print(positions1)
            minDist = 9999999999
            posIndex = 0

            #arclen = cv2.arcLength(sorted_contours[0], True)
            #sorted_contours[0] = cv2.convexHull(sorted_contours[0])
            #sorted_contours[0] = cv2.approxPolyDP(sorted_contours[0], 0.9*arclen, True)
            for contourIndex in range(0, len(sorted_contours[0])):

                currentContour = sorted_contours[0][contourIndex]
                print("current contour")
                print(currentContour)

                dist = (currentContour[0][0] - positions1[0][0])**2 + (currentContour[0][1] - positions1[0][1])**2

                if dist < minDist:
                    minDist = dist
                    posIndex = contourIndex
            #posIndex = np.where(abs(sorted_contours[0] - positions1) == np.nanmin(abs(sorted_contours[0] - positions1)))
            print("pos index = " + str(posIndex))
            positions1 = sorted_contours[0][posIndex][:]
            imgW, imgH = differenceImages[i].shape
            
            positions1[0][0] = r[0] + positions1[0][0] 
            positions1[0][1] = r[1] - positions1[0][1]

            print("positions 1 ")
            print(positions1)

            if len(firstImgPositions) > 0:
                #firstImgPositions = firstImgPositions + [positions1]
                firstImgPositions = np.append(firstImgPositions, positions1, axis = 0)
                print("first img positions")
                print(firstImgPositions)
            else:
                firstImgPositions = np.array(positions1)
                print("current firstImgPositions")
                print(firstImgPositions)
            
        else:

            img2 = scaledBox
            fig, ax = plt.subplots(constrained_layout=True)
            ax.imshow(img2, cmap="gray")
            klicker = clicker(ax, ["event"], markers=["x"])
            plt.show()

            positions2 = klicker.get_positions()
            positions2 = np.array(positions2["event"], dtype = np.float32)

            minDist = 9999999999
            posIndex = 0
            #arclen = cv2.arcLength(sorted_contours[0], True)
            #sorted_contours[0] = cv2.approxPolyDP(sorted_contours[0], 0.9*arclen, True)
            print("sorted contour")
            print(sorted_contours)

            if (len(sorted_contours) > 0):

                for contourIndex in range(0, len(sorted_contours[0])):
                    currentContour = sorted_contours[0][contourIndex]
                    print("current contour")
                    print(currentContour)

                    dist = (currentContour[0][0] - positions2[0][0])**2 + (currentContour[0][1] - positions2[0][1])**2

                    if dist < minDist:
                        minDist = dist
                        posIndex = contourIndex

                positions2 = sorted_contours[0][posIndex][:]
            
            else:
                print("0 length contour")
                print(sorted_contours)
                for contourIndex in range(0, len(sorted_contours)):
                    currentContour = sorted_contours[contourIndex]
                    print("current contour")
                    print(currentContour)

                    dist = (currentContour[0][0] - positions2[0][0])**2 + (currentContour[0][1] - positions2[0][1])**2

                    if dist < minDist:
                        minDist = dist
                        posIndex = contourIndex
                print("current contour")
                print(currentContour)
                print("sorted contour")
                print(sorted_contours)

                positions2 = sorted_contours[posIndex][:]

            imgW, imgH = differenceImages[i].shape
            #r[1] = int(imgW) - int(r[1])
            print("positions 1 =")
            print(positions1)
            print("current r = " + str(r))
            print("x val added = " + str(positions1[0][0]))
            print(" r x contribution = " + str(r[1]))
            print("y val added = " + str(positions1[0][1]))
            print(" r y contribution = " + str(r[0]))
            #currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            
            positions2[0][0] = r[0] + positions2[0][0] 
            positions2[0][1] = r[1] + positions2[0][1]

            if len(newImgPositions) > 0:
                
                newImgPositions = np.append(newImgPositions, positions2, axis = 0)
            else:
   
                newImgPositions = np.array(positions2)
            print("new image positions")
            print(newImgPositions)


        if i != 0 and x == 3:

            des1 = np.array(firstImgPositions, dtype = np.float32)
            des2 = np.array(newImgPositions, dtype = np.float32)
            plt.imshow(intensityImages[i], cmap="gray", vmin = 10, vmax = 30)
            plt.plot(des2[:,0], des2[:, 1], "*")
            plt.show()
            print("initial corner positions")
            print(des1)
            print("final corner positions")
            print(des2)
    
            #M, _ = cv2.estimateAffinePartial2D(des1, des2)
            #M = cv2.getPerspectiveTransform(des1, des2, cv2.DECOMP_LU) #set as default - didn't seem to make much difference


            #M[0, 2] = 0
            #M[1, 2] = 0
            #M[0, 1] = 0
            #M[1, 0] = 0


            print("points used")
            print(des1)
            print("second points")
            print(des2)


            w,h = img1.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            img2 = np.array(intensityImages[i], dtype = np.float32)
            #print("image vales before anyting happens")
            #print(img2)
            #print("values before variable asignment")
            #print(intensityImages[i])
            img2 = (img2-np.nanmin(img2))/(np.nanmax(img2)-np.nanmin(img2)) #renormalises - after contrast adjust so shows
            img2 = img2 * 255
            img2 = np.clip(img2, 0, 255)
            

            img2 = np.array(img2, dtype = np.float32)
            #M = np.array(M, dtype = np.float32)

            #im2 = cv2.warpPerspective(img2, M, (h, w))
            w, h = differenceImages[0].shape
            #im2 = cv2.warpAffine(intensityImages[i], M, (w,h))
            #im2 = cv2.warpPerspective(img, M, (w, h))
            m = cv2.estimateAffinePartial2D(des2, des1)
            m = m[0]
            print("removed m parts")
            print(m[0][1])
            print(m[1][0])
            #remove any rotating
            m[0][1] = 0
            m[1][0] = 0
            print("m = ")
            print(m)
            im2 = cv2.warpAffine(differenceImages[i], (m), (w,h), flags = cv2.INTER_CUBIC)
            
            correctedDifferenceImages = correctedDifferenceImages + [im2]

            im2 = cv2.warpAffine(intensityImages[i], (m), (w, h), flags=cv2.INTER_CUBIC)
            correctedIntensityImages = correctedIntensityImages + [im2]


plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)


img1 = correctedDifferenceImages[0]
img2 = correctedIntensityImages[0]

fig1 = ax1.imshow(img1, cmap="gray")
fig2 = ax2.imshow(img2, cmap="gray", vmin = 10, vmax = 30)

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferenceImages[sb.val]
    img2 = correctedIntensityImages[sb.val]

    fig1.set_data(img1)
        
    plt.draw()

ax1.set_title("differences")
ax2.set_title("intensities")

sb.on_changed(update)
plt.show()

for i in range(0, len(correctedDifferenceImages)):
    print("i = " + str(i))
    #sk_imsave("finalCross/alignedCrossNewFinalFinalFinal_D_%s.tif" % str(i), correctedDifferenceImages[i])
    #sk_imsave("finalCross/alignedCrossNewFinalFinalFinal_I_%s.tif" % str(i), correctedIntensityImages[i])
    sk_imsave("finalCross/alignedCrossReplacement_D_%s.tif" % str(i), correctedDifferenceImages[i])
    sk_imsave("finalCross/alignedCrossReplacement_I_%s.tif" % str(i), correctedIntensityImages[i])