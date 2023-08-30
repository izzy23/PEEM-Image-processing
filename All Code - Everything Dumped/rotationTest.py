import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

def averageImages(imStack):

        imageSum = imStack[0]

        #adding pixel vals for each image
        for image in range(1, len(imStack)):
            
            imageSum = imageSum.astype(float) + imStack[image].astype(float)

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

        #adjusting contrast - so differences can be picked up, and aren't lost when converted to 8-bit later
        #these values worked
        #displayMin = 0.001
        #displayMax = 0.1
        #MAY NEED TO CHANGE FOR OTHER STACKS - NOT SURE
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
    #may need with template matching for bigger drift
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
                                     qualityLevel=0.0001,
                                     minDistance=30,
                                     blockSize=3)

        if i+1 < len(imgArr):
            current = imgArr[i+1]
            current = current *255
            current = np.array(current, dtype="uint8")
 
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prevImg, current, prev_pts, None) 
 
        # Filter only valid points
        #discards points if nt found in both images (i.e. if brightness change too drastic)
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        print("no prev points")
        print(len(prev_pts))
        print("no current points")
        print(len(curr_pts))
 
        #Find transformation matrix
        #m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        print("m")
        print(m)
 
        # Extract traslation
        dx = m[0][0][2]
        dy = m[0][1][2]
        
        # Extract rotation angle
        #da = np.arctan2(m[1,0], m[0,0])
        da = np.arctan2(m[0][1][0], m[0][0][0])
 
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

        print(len(frames))
  
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]
 
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        print("dx = " + str(dx))
        print("dy = " + str(dy))
        print("theta = " + str(da))
 
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(current, m, (w,h))

        frames = frames + [frame_stabilized]
    
    plt.imshow(frame_stabilized)
    plt.show()
    return frames

imageNumbers = ["321124-321125", "321127-321136", "321457-321466", "321467-321476", "321527-321536", "321537-321546", "321617-321626", "321627-321636"]
differenceImages = []
intensityImages = []

for i in imageNumbers: 
    #RC294 - Cross Centre\321467-321476_Diffference.tif

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]


#correctedDiff = interpolateImages(differenceImages)
#correctedInt = interpolateImages(intensityImages)

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []
initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    print("differenceMin = " + str(np.nanmin(image)))
    print("differenceMax = " + str(np.nanmax(image)))
    avg = np.nanmean(image)
    std = np.nanstd(image)
    #np.clip(image, 200, 250)
    #newImage = avg + ((image - initialAvg) * (std / initialStd))
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    np.clip(newImage, -1, 1)
    #np.clip(newImage, -0.5, 0.5)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    image = cv2.medianBlur(image, 5)

    imageStack = imageStack + [image]

correctedDiff = imageStack


initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []


for image in intensityImages:
    print("looping")
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #print("initial mean " + str(avg))
    #print("initial standard deviation " + str(std))
    image = np.clip(image, 5, 40)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))


    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
correctedInt = imageStack
print("length before corrections")
print(len(intensityImages))


differenceImages = np.array(differenceImages, dtype=np.float32)
intensityImages = np.array(intensityImages, dtype=np.float32)


correctedDiff = driftCorrect(correctedDiff)
correctedInt = driftCorrect(correctedInt)

plt.axis("off")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

img1 = correctedDiff[0]
img2 = correctedInt[0]

fig1 = ax1.imshow(img1, cmap="gray")
fig2 = ax2.imshow(img2, cmap="gray")


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 10, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDiff[sb.val]
    img2 = correctedInt[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)

    plt.draw()

avgIntensity = averageImages(correctedInt)

avgDiff = averageImages(correctedDiff)

ax3.imshow(avgDiff, cmap="gray")
ax4.imshow(avgIntensity, cmap="gray")

ax1.set_title("Diff")
ax2.set_title("Intensity")
sb.on_changed(update)
plt.show()
