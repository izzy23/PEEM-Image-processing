import numpy as np
import h5py
import cv2
import time
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
import skimage
from skimage import io


def imgDiff(images1, images2):
    #print("input img array shape")
    #print(images.shape)
    avgE1 = images1
    avgE2 = images2

    #difference = (avgE1.astype(float) - avgE2.astype(float)) / (avgE1.astype(float) + avgE2.astype(float))
    difference = (avgE1 - avgE2) / (avgE1 + avgE2)

    #for xIndex in range(0, len(avgE1[:][0])):
    #    for yIndex in range(0, len(avgE1[0][:])):
    #        if avgE1[xIndex][yIndex] == -avgE2[xIndex][yIndex]:
    #            difference[xIndex][yIndex] = 2 * avgE1(avgE1[xIndex][yIndex])

    return difference

def averageImages(imStack):

    imageSum = imStack[0]

    #adding pixel vals for each image
    for image in range(1, len(imStack)):
        
        #can't remove float - 8bit wrap around weird thing
        imageSum = imageSum.astype(float) + imStack[image].astype(float)
        #imageSum = imageSum + imStack[image]

    #becomes average
    imageSum = imageSum / len(imStack)

    return imageSum

def fixBorder(img):
  s = img.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  img = cv2.warpAffine(img, T, (s[1], s[0]))
  
  return img

def interpolateImages(imgArray):

    interpolatedImages = []
    
    #for currentImg in imgArray:
        
        #img = cv2.imread(currentImg)
        #normalise - so all pixel values less than 1
    #    currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        #currentImg = currentImg / np.max(currentImg)
        
    #    img = currentImg

        #changed fx and fy = 10 to 2 so maybe quicker for now
        #This gives sub-pixel resolution
        #inter linear is just linear, can switch - LINEAR RECOMENDED BY WEBSITE
    #    scale = 1
    #    interpolatedImage = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_LANCZOS4)
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 6, fy = 6, interpolation = cv2.INTER_CUBIC)
   
    #    interpolatedImages = interpolatedImages + [interpolatedImage]
    overallMax = 0
    overallMin = 99999999
    meanVals = []
    
    for currentImg in imgArray:
        
        std = np.std(currentImg)
        mean = np.mean(currentImg)

        meanVals = meanVals + [mean]
        currentImg = (currentImg-mean)/(std)
        #scale = 1
        interpolatedImage = currentImg
        
        #interpolatedImage = cv2.resize(currentImg, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        interpolatedImage = interpolatedImage * 255
        interpolatedImage = np.array(interpolatedImage, dtype = np.float32)
        interpolatedImages = interpolatedImages + [interpolatedImage]
    print("mean vals")
    print(meanVals)

    #print(len(interpolatedImages))

    #then normalise based on max and min of entire image array
    #for currentImg in imgArray:
    #    currentImg = (currentImg-overallMin)/(overallMax-overallMin)
    #    scale = 1
    #    interpolatedImage = cv2.resize(currentImg, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
    #    interpolatedImages = interpolatedImages + [interpolatedImage]

    
    return np.array(interpolatedImages, dtype = np.float32)

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    # The larger the more stable the video, but less reactive to sudden panning
    # may need with template matching for bigger drift
    SMOOTHING_RADIUS = 800    
    smoothed_trajectory = np.copy(trajectory)

    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory

def driftCorrect(imgArr):

    #can't do to whole array at once - runs out of memory
    #imgArr = imgArr * 255
    #imgArr = np.array(imgArr, dtype="uint8")

    #converts to RGB so openCV can read
    firstImg = imgArr[0] * 255
    firstImg = np.array(firstImg, dtype="uint8")

    w, h = firstImg.shape

    # Pre-define transformation-store array
    transforms = np.zeros((len(imgArr)-1, 3), np.float32) 
    prevImg = firstImg
    
    #REMOVED -2
    for i in range(0, len(imgArr)-1):
        print("entered loop")
        # Detect feature points in previous frame
        #quality level was initially 0.1
        prev_pts = cv2.goodFeaturesToTrack(firstImg,
                                     maxCorners=200,
                                     qualityLevel=0.1,
                                     minDistance=10,
                                     blockSize=3)
        # Convert to grayscale
        #print("points detected")
        #print(prev_pts)

        #didn't work - weird datatype
        #curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        
        current = imgArr[i+1]
        current = current * 255
            
        #current = np.array(current, dtype="uint8")
        current = current.astype("uint8")
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prevImg, current, prev_pts, None, maxLevel=1) 
       
 
        # Filter only valid points
        #discards points if nt found in both images (i.e. if brightness change too drastic)
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
 
        #Find transformation matrix
        #m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
 
        # Extract traslation
        dx = m[0][0][2]
        dy = m[0][1][2]
        
        # Extract rotation angle
        #da = np.arctan2(m[1,0], m[0,0])
        #da = np.arctan2(m[0][1][0], m[0][0][0])
        da = 0
        print("dx = " + str(dx))
        print("dy = "+ str(dy))
 
        # Store transformation
        transforms[i] = [dx,dy,da]
 
        # Move to next frame
        prevImg = current
    
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory) #CHECK THIS LINE

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
 
    # Calculate newer transformation array

    transforms_smooth = transforms + difference

    current = imgArr[0]
    current = current *255
    current = np.array(current, dtype="uint8")
    
    frames = [current]

    #REMOVED -2

    for i in range(0, len(imgArr) - 1):
        current = imgArr[i+1]
        current = current *255
        current = np.array(current, dtype="uint8")

        #print(len(frames))
  
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]
        #print("calculated angle change")
        #print(da)
 
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)

        m[0, 0] = 1
        m[1, 1] = 1
        m[0,2] = dx
        m[1,2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(current, m, (w,h))

        frames = frames + [frame_stabilized]

    return frames

imageNumbers = np.arange(321128, 321137)
differenceImages = [io.imread(r"writeImages\321127_D_avg.tif").astype(np.float32)]
intensityImages = [io.imread(r"writeImages\321127_I_avg.tif").astype(np.float32)]

for i in imageNumbers: 
    differenceImg = io.imread(r"writeImages\%s_D_avg.tif" % str(i)).astype(np.float32)
    intensityImg = io.imread(r"writeImages\%s_I_avg.tif" % str(i)).astype(np.float32)

    #avg = np.mean(differenceImg)
    #std = np.std(differenceImg)
    #differenceImg = (differenceImg - avg)/std
    #differenceImg = (differenceImg-np.nanmin(differenceImg))/(np.nanmax(differenceImg)-np.nanmin(differenceImg))
    #intensityImg = (intensityImg-np.nanmin(intensityImg))/(np.nanmax(intensityImg)-np.nanmin(intensityImg))


    #plt.imshow(differenceImg, cmap="gray", vmin = -0.1, vmax=0.1)
    #plt.show()
    #plt.imshow(intensityImg, cmap="gray", vmin = 0, vmax=25)
    #plt.show()
    differenceImages = differenceImages + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

newStack = []
min = np.nanmin(intensityImages[0])
max = np.nanmax(intensityImages[0])

initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])

for image in intensityImages:
    image = cv2.medianBlur(image, 5)
    
    avg = np.nanmean(image)
    std = np.nanstd(image)

    print("initial mean " + str(avg))
    print("initial standard deviation " + str(std))
    image = np.clip(image, 0, 30)

    #newImage = avg + ((image - initialAvg) * (std / initialStd))
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, 0, 255)
    #newImage = (newImage-np.nanmin(newImage))/(np.nanmax(newImage)-np.nanmin(newImage))
    #newImage = newImage * 255
    avg = np.nanmean(newImage)
    std = np.nanstd(newImage)
    
    print("final mean" + str(avg))
    print("final standard deviation " + str(std))

    
    newStack = newStack + [newImage]

intensityImages = newStack


#for image in intensityImages:

    #print("initial min and max")
    #print(np.nanmin(image))
    #print(np.nanmax(image))
    #avg = np.nanmean(image)
    #std = np.nanstd(image)
    #image = (image - avg) / std
    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    
    #image = (image - min) / (max - min)
    #image = image * 2
    #image = image - 1
    
    #image = np.array(image * 255, dtype = "uint8")
    #image = cv2.equalizeHist(image)
    #image = np.clip(image, 5, 220)

    #image = image + 1
    #clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8,8))
    #image = clahe.apply(image)
    
    #image = cv2.Canny(image = image, threshold1 = 150, threshold2 = 160)
    #image = cv2.Canny(image = image, threshold1 = 100, threshold2 = 160)
    #image = np.array(image, dtype = np.float32)
    #plt.imshow(image)
    #plt.show()
    #image = np.array((image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)), dtype = np.float32)
    #image = image * 255

    #print("new min and max")
    #print(np.nanmin(image))
    #print(np.nanmax(image))
    #print("avg = " + str(avg))
    #print("std = " + str(std))

    #newStack = newStack + [image]
#intensityImages = newStack

#intenistyImages = np.array(newStack, dtype = np.float32)
print("no of intensity images")
print(len(newStack))
plt.imshow(intensityImages[5])
plt.show()

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

newStack = []
for image in differenceImages:
    image = cv2.medianBlur(image, 5)

    avg = np.nanmean(image)
    std = np.nanstd(image)

    #newImage = avg + ((image - initialAvg) * (std / initialStd))
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    
    #newImage = (newImage-np.nanmin(newImage))/(np.nanmax(newImage)-np.nanmin(newImage))
    #newImage = newImage * 255
    avg = np.nanmean(newImage)
    std = np.nanstd(newImage)
    image = newImage


    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    #image = image * 2
    #image = image - 1
    #image = np.array(image, dtype = np.float32)

    newStack = newStack + [image]

differenceImages = np.array(newStack, dtype = np.float32)

plt.imshow(intensityImages[0], cmap="gray")
plt.show()
#run drift correction on averaged stacks
#differenceImages = interpolateImages(differenceImages)
#intensityImages = interpolateImages(intensityImages)

correctedDifferenceImages = driftCorrect(differenceImages)
correctedIntensityImages = driftCorrect(intensityImages)

differenceAverage = averageImages(correctedDifferenceImages)
intensityAverage = averageImages(correctedIntensityImages)

plt.axis("off")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

ax1.imshow(differenceAverage, cmap="gray")
ax2.imshow(intensityAverage, cmap="gray")

fig1 = ax3.imshow(differenceImages[0], cmap="gray", vmin = -0.1, vmax = 0.1)
fig2 = ax4.imshow(intensityImages[0], cmap="gray", vmin = 10, vmax = 35)

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = differenceImages[sb.val]
    img2 = intensityImages[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

sb.on_changed(update)
plt.show()
