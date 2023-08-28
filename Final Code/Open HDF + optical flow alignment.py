import numpy as np
import h5py
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d
from skimage.io import imsave as sk_imsave

def imgDiff(images1, images2):
    #images at 2 different energies
    avgE1 = images1
    avgE2 = images2

    #sometimes needed if wrap around issues
    #difference = (avgE1.astype(float) - avgE2.astype(float)) / (avgE1.astype(float) + avgE2.astype(float))
    
    #to find the difference image, weighted by total
    difference = (avgE1 - avgE2) / (avgE1 + avgE2)

    return difference

def averageImages(imStack):

    imageSum = imStack[0]

    #adding pixel vals for each image
    for image in range(1, len(imStack)):
        
        #can't remove float - 8bit wrap around weird thing
        imageSum = imageSum.astype(float) + imStack[image].astype(float)

    #becomes average
    imageAvg = imageSum / len(imStack)

    return imageAvg

def interpolateImages(imgArray):
    interpolatedImages = []

    for currentImg in imgArray:
        
        #normalise pixel values
        currentImg = (currentImg-np.min(currentImg))/(np.max(currentImg)-np.min(currentImg))
        
        
        img = currentImg

        #scale is used for interpolation. If scale = 3, then final image will have 3x as many pixels.
        #linear interpolation semes to work well, and is reccomended as standard by openCV website
        scale = 3
        interpolatedImage = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)

        interpolatedImages = interpolatedImages + [interpolatedImage]
    
    return interpolatedImages

def movingAverage(curve, radius):
    #reigon considered
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
    #larger smoothing radius is better for small movements,
    #smaller smoothing radius better for large motion between images
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

    # Pre-define to store dx, dy, and da (angle change)
    transforms = np.zeros((len(imgArr)-1, 3), np.float32) 
    prevImg = firstImg
    
    #Loops through array of images
    for i in range(0, len(imgArr)-1):
        # Detect features in initial image
        prev_pts = cv2.goodFeaturesToTrack(firstImg,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
        
        # checks if enough uncorrected images remain.
        if i+1 < len(imgArr):
            current = imgArr[i+1]
            current = current *255
            current = np.array(current, dtype="uint8")
 
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prevImg, current, prev_pts, None, maxLevel=1) 
 
        # Removes invalid points (i.e. if only found in one image / drastic brihtenss change / etc)
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
 
        #Find transformation matrix 
        #affine transform allows for rotation, translation, stretching,and shearing.
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
 
        # Found translations and rotation angles
        dx = m[0][0][2]
        dy = m[0][1][2]
        da = 0
 
        # Store transformation
        transforms[i] = [dx,dy,da]
 
        # Move to next image
        prevImg = current
    
    # Ensures translations are reasonable - should follow a realistic path if camera drift - i.e. not random
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory) 

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
 
    # New transformation array - should be moved closer to a realistic path.
    transforms_smooth = transforms + difference

    current = imgArr[0]
    current = current *255
    current = np.array(current, dtype="uint8")
    
    images = [current]

    #REMOVED -2

    for i in range(0, len(imgArr) - 1):
        current = imgArr[i+1]
        current = current *255
        current = np.array(current, dtype="uint8")

        # New transformations - after trajectory smoothing
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]

        # Reconstruct affine transform matrix
        m = np.zeros((2,3), np.float32)

        #uncomment to allow for rotations - doens't seem to be needed here.
        #m[0,0] = np.cos(da)
        #m[0,1] = -np.sin(da)
        #m[1,0] = np.sin(da)
        #m[1,1] = np.cos(da)

        #set 1 for 0 rotation
        m[0, 0] = 1
        m[1, 1] = 1

        #x and y motion
        m[0,2] = dx
        m[1,2] = dy
 
        # Apply affine transform to image
        finalImage = cv2.warpAffine(current, m, (w,h))

        images = images + [finalImage]


    return images


# open the file as 'f'
with h5py.File('Diamond images\medipix-321127.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]
    data = np.array(f[key]["instrument"]["detector"]["data"])   #from file tree

    #array of images from file - at the two different energies
    initialStackE1 = data[0][:]
    intitalStackE2 = data[1][:]

    #array of all energies
    entireStack = np.concatenate((initialStackE1, intitalStackE2))
    
    #increases no of pixels in image - better resolution
    entireStack = interpolateImages(entireStack)

    correctedEntireStack = driftCorrect(entireStack)


    plt.axis("off")

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)


    #finds difference between drift corrected averages

    #average images from both energies
    stack1Avg = averageImages(correctedEntireStack[0:20])
    stack2Avg = averageImages(correctedEntireStack[20:40])

    #gives the intensity image
    overallAverage = averageImages([stack1Avg, stack2Avg])

    ax1.imshow(stack1Avg, cmap="gray", vmin = 10, vmax = 30)
    ax2.imshow(stack2Avg, cmap="gray", vmin = 10, vmax = 30)
    ax4.imshow(overallAverage, cmap="gray", vmin = 10, vmax = 30)

    #subtracts for the difference image
    differenceImg = imgDiff(stack1Avg, stack2Avg)
    ax3.imshow(differenceImg, cmap="gray", vmin=-0.1, vmax = 0.1)


    ax1.set_title("average E1")
    ax2.set_title("average E2")
    ax3.set_title("difference")
    ax4.set_title("Intensity")

    plt.show()
    
    sk_imsave("writeImages/321466_D_avg.tif", differenceImg)
    sk_imsave("writeImages/321466_I_avg.tif", overallAverage)

    f.close()

