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

def translateImage(imgH, imgW, img, fixedPos, driftPos):

    #difference in x and y positions for ROI between images
    tx = driftPos[0] - fixedPos[0]
    ty = driftPos[1] - fixedPos[1]

    print("image shifts")
    print("x shift = " + str(tx))
    print("y shift = " + str(ty))

    #generates translation matrix
    translationMatrix = np.array([
    [1, 0, -tx],
    [0, 1, -ty]
    ], dtype=np.float32)

    #apply transformation
    translatedImage = cv2.warpAffine(src=img, M=translationMatrix, dsize=(imgW, imgH))

    return translatedImage


def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    
    return rotatedImg


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

#select corners for zoom in
firstImgPositions = []
correctedDifferenceImages = []
correctedDifferenceImages = correctedDifferenceImages + [differenceImages[0]]
correctedIntensityImages = []

for i in range(0, len(differenceImages)):
    
    newImgPositions = []
    displayImg = np.clip(intensityImages[i], 10, 25)
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))
    #displayImg = intensityImages[i]
    plt.imshow(displayImg, cmap="gray")
    plt.show()
    # Select ROI
    w, h = displayImg.shape
    #loop 4 times to select each corner
    for x in range(0, 4):

        # Naming a window
        cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

        #resizes window, not actual image
        cv2.resizeWindow("select ROI", w , h)

        r = cv2.selectROI("select ROI", displayImg)
        r = np.array(r)
        print("r = " + str(r))

        img = differenceImages[i]
        currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        currentBox = (currentBox-np.nanmin(currentBox))/(np.nanmax(currentBox)-np.nanmin(currentBox))   #rescales intensity for box - don't remove!!
        currentBox = currentBox * 255
        currentBox = np.array(currentBox, dtype="uint8")
        scaledBox = currentBox
        currentBox = cv2.medianBlur(currentBox, 3)
        ret,thresh = cv2.threshold(currentBox,120,255,0)
        currentBox = thresh

        edgeImg = cv2.Canny(currentBox, 0, 200)

        #plt.figure()
        #plt.axis("off")
        #dst = cv2.cornerHarris(currentBox,20,3,0.04)
        #result is dilated for marking the corners, not important
        #dst = cv2.dilate(dst,None)
        contours, higherarchy = cv2.findContours(currentBox, 1, cv2.CHAIN_APPROX_TC89_KCOS)

        while len(contours) == 0:
            # Naming a window
            cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

            #resizes window, not actual image
            cv2.resizeWindow("select ROI", w , h)

            r = cv2.selectROI("select ROI", displayImg)
            r = np.array(r)
            print("r = " + str(r))

            img = differenceImages[i]
            currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

            currentBox = (currentBox-np.nanmin(currentBox))/(np.nanmax(currentBox)-np.nanmin(currentBox))   #rescales intensity for box - don't remove!!
            currentBox = currentBox * 255
            currentBox = np.array(currentBox, dtype="uint8")
            scaledBox = currentBox
            currentBox = cv2.medianBlur(currentBox, 3)
            ret,thresh = cv2.threshold(currentBox,120,255,0)
            currentBox = thresh

            edgeImg = cv2.Canny(currentBox, 0, 200)

            contours, higherarchy = cv2.findContours(currentBox, 1, cv2.CHAIN_APPROX_TC89_KCOS)
        plt.figure()
        plt.axis("off")
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(scaledBox, sorted_contours, 0, (0,255,0), 1)

        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        #edgeImg[dst>0.2*dst.max()]=[255]
        ax2.imshow(currentBox, cmap = "gray")
        ax3.imshow(scaledBox, cmap="gray")
        ax1.imshow(edgeImg)

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
            #print("selected position = " )
            #print(positions1)
            minDist = 9999999999
            posIndex = 0
            for contourIndex in range(0, len(sorted_contours[0])):
                currentContour = sorted_contours[0][contourIndex]

                dist = (currentContour[0][0] - positions1[0][0])**2 + (currentContour[0][1] - positions1[0][1])**2

                if dist < minDist:
                    minDist = dist
                    posIndex = contourIndex
            #posIndex = np.where(abs(sorted_contours[0] - positions1) == np.nanmin(abs(sorted_contours[0] - positions1)))
            print("pos index = " + str(posIndex))
            positions1 = sorted_contours[0][posIndex][:]
            imgW, imgH = differenceImages[i].shape

            #r[1] = int(imgW) - int(r[1])


            print("current r = " + str(r))
            print("x val added = " + str(positions1[0][0]))
            print(" r x contribution = " + str(r[1]))
            print("y val added = " + str(positions1[0][1]))
            print(" r y contribution = " + str(r[0]))
            #currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            
            positions1[0][0] = r[0] + positions1[0][0] 
            positions1[0][1] = r[1] - positions1[0][1]


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
            for contourIndex in range(0, len(sorted_contours[0])):
                currentContour = sorted_contours[0][contourIndex]

                dist = (currentContour[0][0] - positions2[0][0])**2 + (currentContour[0][1] - positions2[0][1])**2

                if dist < minDist:
                    minDist = dist
                    posIndex = contourIndex

            positions2 = sorted_contours[0][posIndex][:]

            imgW, imgH = differenceImages[i].shape

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
            plt.imshow(intensityImages[i], cmap="gray", vmin = 5, vmax = 30)
            plt.plot(des2[:,0], des2[:, 1], "*")
            plt.show()
            print("new image translate")

            position1Sum = 0
            position2Sum= 0
            scaleSum = 0
            dxSum = 0
            dySum = 0

            for k in range(0, 2):
                #translate to match corner
                position1Sum = position1Sum + des1[k*2]
                position2Sum = position2Sum + des2[k*2]
                h, w = differenceImages[i].shape

                print("first positions")
                print(des1)
                #print("hopefully first point")
                #print(position1)
                
                position1 = des1[(k*2)]
                position2 = des2[(k*2)]
                position1b = des1[(k*2) + 1]
                position2b = des2[(k*2) + 1]

                position1x = position1[0]
                position1y = position1[1]
                position1bx = position1b[0]
                position1by = position1b[1]

                position2x = position2[0]
                position2y = position2[1]
                position2bx = position2b[0]
                position2by = position2b[1]
            
                #scale = float(np.mean((position1b - position1) / (position2b - position2)))
                #scaleSum = float(scaleSum + scale)
                print("des1 = " + str(des1[k*2]))
                print("p1bx = " + str(position1bx))
                print("p1x = " + str(position1x))
                xScale = (position1bx - position1x) / (position2bx - position2x)
                yScale = (position1by - position1y) / (position2by - position2y)
                scale = np.mean([xScale, yScale])

                dx = np.mean([position1bx - position2bx, position1x - position2x])
                dy = np.mean([position1by - position2by, position1y - position2y])
                dxSum = dxSum + dx
                dySum = dySum + dy
                print("dx = " + str(dx))
                print("dy = " + str(dy))

                print("x scale = " + str(xScale))
                print("y scale = " + str(yScale))
                print("overall scale = " + str(scale))
                scaleSum = scaleSum + scale

            position1 = position1Sum / 2    #leaves as [x, y]
            position2 = position2Sum / 2
            print("position 1 = " + str(position1))
            print("position 2 = " + str(position2))
            h, w = differenceImages[i].shape
            scale = scaleSum / 2
            #position1 = np.round(position1 * scale)
            #position2 = np.round(position2 * scale)
            position1 = np.array([0, 0], dtype = np.int)
            position2 = np.array([dxSum/2, dySum/2], dtype = np.float64)
            position2 = position2 * scale
            scaleH = ((h * scale) - h) / 2
            scaleW = ((w * scale) - w) / 2
            position2[0] = position2[0] + scaleH
            position2[1] = position2[1] + scaleW
            position2 = np.round(abs(position2))
            position2 = np.array(position2, dtype = np.int)
            print("pos 2 = " + str(position2))

            #zoom to match corners

            resizedImg = cv2.resize(differenceImages[i], (int(round(h * scale)), int(round(w * scale))), fx=scale, fy=scale, interpolation = cv2.INTER_LANCZOS4)
            print("showing resized image")
            plt.imshow(resizedImg, cmap="gray")
            plt.show()
            
            translatedImage = translateImage(h, w, resizedImg, position1, position2)
            translatedImage = translatedImage[0:h, 0:w]
            correctedDifferenceImages = correctedDifferenceImages + [translatedImage]


            #M = np.array(M, dtype = np.float32)
            
            #im2 = cv2.warpPerspective(img2, M, (h, w))
            #w, h = differenceImages[0].shape
            #im2 = cv2.warpAffine(intensityImages[i], M, (w,h))
            #im2 = cv2.warpPerspective(img, M, (w, h))

            #correctedDifferenceImages = correctedDifferenceImages + [im2]


plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)

img1 = correctedDifferenceImages[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = -0.08, vmax = 0.1)

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