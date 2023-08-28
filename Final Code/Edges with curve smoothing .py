import numpy as np
import h5py
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave
from mpl_point_clicker import clicker
from matplotlib.widgets import RadioButtons
from math import sqrt


def angle(vectors):
    
    #Returns the angles between vectors.

    vector2 = vectors[1:]
    vector1=vectors[:-1]
    return np.arccos((vector1*vector2).sum(axis=1)/(
        np.sqrt((vector1**2).sum(axis=1)*(vector2**2).sum(axis=1))))

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
    #da = rotation angle
    
    h, w = img.shape
    center = (w / 2, h / 2)     #centre of image

    #so no resizing
    scale = 1

    #finds affine transform matrix needed to rotate image about its centre
    m = cv2.getRotationMatrix2D(center, da, scale)

    #applies afine transform
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    
    return rotatedImg

def contourPrep(r, img, lowBinaryThreshold, highBinaryThreshold):
    #binarises cropped image based on low binary threshold.
    #black = pixels below  low binary threshold
    #white = all pixels above low binary threshold

    #cropps to user selected ROI
    currentBox = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    #readjusts contrast of just ROI so features are noticable
    currentBox = (currentBox-np.nanmin(currentBox))/(np.nanmax(currentBox)-np.nanmin(currentBox))
    currentBox = currentBox * 255
    currentBox = np.array(currentBox, dtype="uint8") #openCV needs this datatype

    #output image saved - to see with adjusted contrast
    scaledBox = currentBox

    #to limit effect of dead pixels, etc
    currentBox = cv2.medianBlur(currentBox, 3)

    #binarises image - convert to just black or white pixels
    ret,thresh = cv2.threshold(currentBox,lowBinaryThreshold,highBinaryThreshold,0) #120, 255, 0
    currentBox = thresh

    #used to display all edges found in binarised image
    edgeImg = cv2.Canny(currentBox, 0, 200)

    #finds edge lines within binarised image.  Stores arrays of points for all edges.
    contours, higherarchy = cv2.findContours(currentBox, 1, cv2.CHAIN_APPROX_TC89_KCOS)

    #Longest contours will be at index 0.  Smallest contours will be furthest from 0.
    #Longest contour should be main edge visible in image.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return currentBox, scaledBox, edgeImg, sorted_contours

imageNumbers = ["321124-321125", "321127-321136", "321227-321236", "321457-321466", "321527-321536", "321617-321626", "321627-321636"]

differenceImages = []
intensityImages = []

for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))
    #plt.imshow(differenceImg, cmap="gray", vmin = -0.08, vmax = 0.1)
    #plt.show()

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

#rotates images manually

offsetAngle = 50

angles = [0, 0, -30, -45, -50, -50, -90]
angles = np.array(angles, dtype = np.int)
angles = angles + offsetAngle

for i in range(0, len(intensityImages)):

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

#select corners for zoom in

firstImgPositions = [] #used to align 

#arrays of aligned images
correctedDifferenceImages = []
correctedIntensityImages = []

for i in range(0, len(differenceImages)):
    
    newImgPositions = [] #for current image in loop

    #adjust contrast so image can be seen
    displayImg = np.clip(intensityImages[i], 10, 30)
    displayImg = (displayImg-np.nanmin(displayImg))/(np.nanmax(displayImg)-np.nanmin(displayImg))

    # Select ROI - 4 corners in cross
    w, h = displayImg.shape

    #loop to select each corner
    for x in range(0, 4):
        #default values for binarising
        lowBinaryThreshold = 120
        highBinaryThreshold = 255
        
        # Naming a window
        cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

        #resizes window, not actual image
        cv2.resizeWindow("select ROI", w , h)

        #gets box dimentions
        r = cv2.selectROI("select ROI", displayImg)
        r = np.array(r)

        #gets edge images and contours
        currentBox, scaledBox, edgeImg, sorted_contours = contourPrep(r, differenceImages[i], 120, 255)
        

        while len(sorted_contours) < 1:
            #user reselects ROI if no contours found in image - stops program crashing later

            # Naming a window
            cv2.namedWindow("select ROI", cv2.WINDOW_NORMAL)

            #resizes window, not actual image
            cv2.resizeWindow("select ROI", w , h)

            #gets box dimentions
            r = cv2.selectROI("select ROI", displayImg)
            r = np.array(r)

            #gets edge images and contours
            currentBox, scaledBox, edgeImg, sorted_contours = contourPrep(r, differenceImages[i], 120, 255)
        
        
        linePoints = []
        first = True

        for point in sorted_contours[0]:
            
            if point[0, 0] != 0 and point[0, 1] != 0 and point[0, 0] != w and point[0, 1] != h:
                
                linePoints.append((int(point[0, 0]), int(point[0, 1])))

        #to find smoothed trajectory
        simplified_trajectory = rdp(linePoints, epsilon = 3) #original epsilon = 200

        #x and y points on curve
        sx = [t[0] for t in simplified_trajectory]
        sy = [t[1] for t in simplified_trajectory]
        sx = np.array(sx, dtype = np.int)
        sy = np.array(sy, dtype = np.int)


        # Minimum angle to consider as a turning point
        min_angle = np.pi / 3.0

        # Direction vectors for simplified trajectory
        directions = np.diff(simplified_trajectory, axis=0)
        theta = angle(directions)

        # finds points with dramatic angle changes
        idx = np.where(theta > min_angle)[0] + 1
        idx = np.array(idx, dtype = np.int)
        turningPointsX = sx[idx]
        turningPointsY = sy[idx]
        turningPoints = []

        #extract likely turning points
        for j in range(0, len(turningPointsX)):
            x = turningPointsX[j]
            y = turningPointsY[j]
            turningPoints = turningPoints + [[x, y]]

        plt.figure()
        plt.axis("off")
        #cv2.drawContours(scaledBox, sorted_contours, 0, (0,255,0), 1)

        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)

        cv2.drawContours(scaledBox, sorted_contours, 0, (0,255,0), 1)

        ax2.imshow(currentBox, cmap = "gray")
        ax3.imshow(255 - scaledBox, cmap="gray")
        ax3.plot(turningPointsX, turningPointsY, "*")
        ax1.imshow(edgeImg)

        rax = plt.axes([0.47, 0.5, 0.1, 0.15])

        #use to manually adjust binarisation cutoff 
        #user changes until good contour found
        radioHandle = RadioButtons(rax, ("lower threshold", "higher threshold"))

        def buttonFunc(label):
            global lowBinaryThreshold
            global highBinaryThreshold
            global r
            global currentBox
            global scaledBox
            global edgeImg
            global sorted_contour
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            #adjust binarisation cutoff
            if label == "lower threshold":
                lowBinaryThreshold = lowBinaryThreshold - 5
            if label == "higher threshold":
                lowBinaryThreshold = lowBinaryThreshold + 5

            #gets contours and edge positions
            currentBox, scaledBox, edgeImg, sorted_contours = contourPrep(r, differenceImages[i], lowBinaryThreshold, highBinaryThreshold)
            linePoints = []

            first = True
            for point in sorted_contours[0]:
                #avoid edge of ROI
                if point[0, 0] != 0 and point[0, 1] != 0 and point[0, 0] != w and point[0, 1] != h:
                    linePoints.append((int(point[0, 0]), int(point[0, 1])))

            #smooths trajectory as before
            simplified_trajectory = rdp(linePoints, epsilon = 3)

            #all x and y poitns
            sx = [t[0] for t in simplified_trajectory]
            sy = [t[1] for t in simplified_trajectory]
            sx = np.array(sx, dtype = np.int)
            sy = np.array(sy, dtype = np.int)


            # Use to find turning points
            min_angle = np.pi / 3.0

            #Find angles of direction changes
            directions = np.diff(simplified_trajectory, axis=0)
            theta = angle(directions)

            # Find significant turning points
            idx = np.where(theta > min_angle)[0] + 1
            idx = np.array(idx, dtype = np.int)

            turningPointsX = sx[idx]
            turningPointsY = sy[idx]
            turningPoints = []

            #extract possible turning points
            for j in range(0, len(turningPointsX)):
                x = turningPointsX[j]
                y = turningPointsY[j]
                turningPoints = turningPoints + [[x, y]]

            cv2.drawContours(scaledBox, sorted_contours, 0, (0,255,0), 1)
            
            ax2.imshow(currentBox, cmap = "gray")
            ax3.imshow(255 - scaledBox, cmap="gray")
            ax3.plot(turningPointsX, turningPointsY, "*")
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

            #user places point near a corner
            klicker = clicker(ax, ["event"], markers=["x"])
            plt.show()

            #gets position of point placed by user
            positions1 = klicker.get_positions()
            positions1 = np.array(positions1["event"], dtype = np.float32)
            
            minDist = 9999999999
            posIndex = 0

            for cornerIndex in range(0, len(turningPoints)):
                corner = turningPoints[cornerIndex]
                dist = (corner[0] - positions1[0][0])**2 + (corner[1] - positions1[0][1])
                if dist < minDist:
                    minDist = dist
                    posIndex = cornerIndex
            imgW, imgH = differenceImages[i].shape

            #possible corner positions within ROI selected
            positions1[0][0] = turningPoints[cornerIndex][0]
            positions1[0][1] = turningPoints[cornerIndex][1]
            
            #Turn ROI corner positions to corner position in overall image
            positions1[0][0] = r[0] + positions1[0][0] 
            positions1[0][1] = r[1] - positions1[0][1]

            if len(firstImgPositions) > 0:

                firstImgPositions = np.append(firstImgPositions, positions1, axis = 0)
                
            else:

                firstImgPositions = np.array(positions1)
            
        else:

            img2 = scaledBox
            fig, ax = plt.subplots(constrained_layout=True)
            ax.imshow(img2, cmap="gray")

            #gets user placed point
            klicker = clicker(ax, ["event"], markers=["x"])
            plt.show()

            #coordinates of user placed point
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

            #corner positions in ROI
            positions2 = sorted_contours[0][posIndex][:]

            imgW, imgH = differenceImages[i].shape

            #Corner positions in overall image
            positions2[0][0] = r[0] + positions2[0][0] 
            positions2[0][1] = r[1] + positions2[0][1]

            #stores corner positions found
            if len(newImgPositions) > 0:

                newImgPositions = np.append(newImgPositions, positions2, axis = 0)
            else:
   
                newImgPositions = np.array(positions2)

        if i != 0 and x == 3:

            des1 = np.array(firstImgPositions, dtype = np.float32)
            des2 = np.array(newImgPositions, dtype = np.float32)

            plt.imshow(intensityImages[i], cmap="gray", vmin = 10, vmax = 30)
            plt.plot(des2[:,0], des2[:, 1], "*")
            plt.show()

            w,h = img1.shape

            img2 = np.array(intensityImages[i], dtype = np.float32)

            img2 = (img2-np.nanmin(img2))/(np.nanmax(img2)-np.nanmin(img2)) #renormalises - after contrast adjust so shows
            img2 = img2 * 255
            img2 = np.clip(img2, 0, 255)
            

            img2 = np.array(img2, dtype = np.float32)

            w, h = differenceImages[0].shape

            #estimates affine transformation
            #uses closest match for the 4 points between first image and current image
            m = cv2.estimateAffinePartial2D(des2, des1)
            m = m[0]
            
            #can remove any rotating
            #m[0][1] = 0
            #m[1][0] = 0

            #applies transformation to align current diffence image
            im2 = cv2.warpAffine(differenceImages[i], (m), (w,h))
            
            correctedDifferenceImages = correctedDifferenceImages + [im2]

            #applies transformation to align current intensity image
            im2 = cv2.warpAffine(intensityImages[i], (m), (w,h))
            correctedIntensityImages = correctedIntensityImages + [im2]

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

img1 = correctedDifferenceImages[0]
img2 = correctedIntensityImages[9]

fig1 = ax1.imshow(img1, cmap="gray")
fig2 = ax2.imshow(img2, cmap="gray")

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferenceImages[sb.val]
    img2 = correctedIntensityImages[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("differences")

sb.on_changed(update)
plt.show()

for i in range(0, len(correctedDifferenceImages)):
    sk_imsave("finalCross/alignedCrossFinal_D_%s.tif" % str(i), correctedDifferenceImages[i])
    sk_imsave("finalCross/alignedCrossFinal_D_%s.tif" % str(i), correctedIntensityImages[i])