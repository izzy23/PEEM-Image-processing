import numpy as np
import h5py
import cv2
#import time
#import io
#import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

from skimage.io import imsave as sk_imsave

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

    for currentImg in imgArray:
        
        #img = cv2.imread(currentImg)
        #normalise - so all pixel values less than 1
        currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        #currentImg = currentImg / np.max(currentImg)
        
        img = currentImg

        #changed fx and fy = 10 to 2 so maybe quicker for now
        #This gives sub-pixel resolution
        #inter linear is just linear, can switch - LINEAR RECOMENDED BY WEBSITE
        scale = 3
        interpolatedImage = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_LANCZOS4)
        #interpolatedImage = cv2.resize(img, (0, 0), fx = 6, fy = 6, interpolation = cv2.INTER_CUBIC)

        #adjusting contrast - so differences can be picked up, and aren't lost when converted to 8-bit later
        #these values worked
        #displayMin = 0.001
        #displayMax = 0.1
        #MAY NEED TO CHANGE FOR OTHER STACKS - NOT SURE
        #commented below out for now - cutting data bad
        #displayMin = 0.043
        #displayMax = 0.098
        #interpolatedImage.clip(displayMin, displayMax, out=interpolatedImage) #hopefully adjust contraast

        #interpolatedImage = (interpolatedImage-np.min(interpolatedImage))/(np.max(interpolatedImage)-np.min(interpolatedImage)) #renormalises - after contrast adjust so shows
        #end of contour adjustment
        interpolatedImages = interpolatedImages + [interpolatedImage]
    
    return interpolatedImages

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
    SMOOTHING_RADIUS = 50    
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
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(firstImg,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
        # Convert to grayscale
        #didn't work - weird datatype
        #curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        if i+1 < len(imgArr):
            current = imgArr[i+1]
            current = current *255
            current = np.array(current, dtype="uint8")
 
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
        #m[0,0] = np.cos(da)
        #m[0,1] = -np.sin(da)
        #m[1,0] = np.sin(da)
        #m[1,1] = np.cos(da)
        m[0, 0] = 1
        m[1, 1] = 1
        m[0,2] = dx
        m[1,2] = dy

        #print("dx = " + str(dx))
        #print("dy = " + str(dy))
        #print("theta = " + str(da))
 
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(current, m, (w,h))

        frames = frames + [frame_stabilized]
 
        # Fix border artifacts
        #removed this bit - stretches around centre so same size.
        #I think will break stuff
        #frame_stabilized = fixBorder(frame_stabilized) 
 

 
  # If the image is too big, resize it.
  #don't think needed
  #if(frame_out.shape[1] &gt; 1920):
    #frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
 
    
    #plt.imshow(frame_stabilized)
    #plt.show()
    return frames

#D:\diamond data\mm33587-1\321227
# open the file as 'f'
#with h5py.File('Diamond images\medipix-321127.hdf', 'r') as f:

with h5py.File(r'D:\diamond data\mm33587-1\medipix-321527.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]
    data = np.array(f[key]["instrument"]["detector"]["data"])

    #array of images from file
    imgArrE1 = data[0][:]
    imgArrE2 = data[1][:]

    initialStackE1 = imgArrE1
    intitalStackE2 = imgArrE2
    #entireStack = imgArrE1 + imgArrE2
    entireStack = np.concatenate((initialStackE1, intitalStackE2))
    entireStackInitial = entireStack
    
    #print("stack shape")
    #print(entireStack.shape)

    #imgArrE1 = interpolateImages(imgArrE1)
    #imgArrE2 = interpolateImages(imgArrE2)
    entireStack = interpolateImages(entireStack)

    #correctedStackE1 = driftCorrect(imgArrE1)
    #correctedStackE2 = driftCorrect(imgArrE2)
    correctedEntireStack = driftCorrect(entireStack)
    #print("corrected entire stack shape")
    #print(correctedEntireStack.shape)

    plt.axis("off")

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)


    #correctedAverageE1 = averageImages(correctedStackE1)
    #correctedAverageE2 = averageImages(correctedStackE2)
    #averageWholeStack = averageImages(correctedEntireStack)

    #doesn't work - new code needs at least 3 images to drift correct
    #drift corrects averages from both energies
    #avgImages = [correctedAverageE1, correctedAverageE2]

    #avgImages = driftCorrect(avgImages)

    #ax1.imshow(correctedAverageE1, cmap="gray")
    #ax2.imshow(correctedAverageE2, cmap="gray")
    
    #finds difference between drift corrected averages

    #print("whole stack average shape")
    #print(averageWholeStack.shape)
    stack1Avg = averageImages(correctedEntireStack[0:20])
    stack2Avg = averageImages(correctedEntireStack[20:40])
    overallAverage = averageImages([stack1Avg, stack2Avg])

    ax1.imshow(stack1Avg, cmap="gray", vmin = 10, vmax = 30)
    ax2.imshow(stack2Avg, cmap="gray", vmin = 10, vmax = 30)
    ax4.imshow(overallAverage, cmap="gray", vmin = 10, vmax = 30)

    #stack1Avg = averageWholeStack[0:19][:][:]
    #stack2Avg = averageWholeStack[20:39][:][:]
    #print("split stack shape")
    #print(stack1Avg.shape)
    differenceImg = imgDiff(stack1Avg, stack2Avg)
    ax3.imshow(differenceImg, cmap="gray", vmin=-0.1, vmax = 0.1)


    ax1.set_title("average E1")
    ax2.set_title("average E2")
    ax3.set_title("difference")
    ax4.set_title("Intensity")

    plt.show()
    
    #sk_imsave("writeImages/321466_D_avg.tif", differenceImg)
    #sk_imsave("writeImages/321466_I_avg.tif", overallAverage)

    f.close()

