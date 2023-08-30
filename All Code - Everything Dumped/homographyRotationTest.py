import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider


MAX_MATCHES = 500
#GOOD_MATCH_PERCENT = 0.15
GOOD_MATCH_PERCENT = 0.6


def alignImages(im1, im2):

  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  

  im1Gray = im1
  im2Gray = im2
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_MATCHES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  #matches.sort(key=lambda x: x.distance, reverse=False)
  matches = sorted(matches, key=lambda x: x.distance)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  print("no points")
  print(len(points1))
  print(len(points2))

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

    print("no of points ")
    print(len(points1))
    print(len(points2))
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  #h = cv2.getPerspectiveTransform(points1, points2)
  #h = np.array(h, dtype = np.float32)
  print("current h = " + str(h))
  

  # Use homography
  height, width = im2.shape
  #height = float(height)
  #width = float(width)
  print("height and width")
  print(height)
  print(width)
  im1 = im1.astype(np.float32)
  #h = h.astype(np.float32)
  if len(h) > 0:
     im1Reg = cv2.warpPerspective(im1, h, (width, height))
  else:
     print("NO H")
     im1Reg = im1


  #im1Reg = cv2.warpPerspective(im1, h)
  
  return im1Reg, h



startNo = 0
endNo = 6

imageNumbers = np.arange(startNo + 1, endNo + 1)

differenceImages = [io.imread(r"rotatedCross\%s_D.tif" % str(startNo)).astype(np.float32)]
intensityImages = [io.imread(r"rotatedCross\%s_I.tif" % str(startNo)).astype(np.float32)]

for i in imageNumbers: 

    differenceImg = io.imread(r"rotatedCross\%s_D.tif" % str(i))
    intensityImg = io.imread(r"rotatedCross\%s_I.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]


#correctedDiff = interpolateImages(differenceImages)
#correctedInt = interpolateImages(intensityImages)

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

initialDiff = (differenceImages[0] + 1) / 2
initialDiff = initialDiff * 255
initialAvg = np.nanmean(initialDiff)
initialStd = np.nanstd(initialDiff)

for image in differenceImages:

 
    image = (image + 1)/2
    image = image*255

    avg = np.nanmean(image)
    std = np.nanstd(image)

    newImage = initialAvg + ((image - avg) * (initialStd / std))

    

    image = np.clip(newImage, 125, 140)

    image = (image-np.min(image))/(np.max(image)-np.min(image))

    image = np.array(newImage, dtype = "uint8")
    print("image shape1")
    print(np.nanmin(newImage))
    print(np.nanmax(newImage))
    #image = cv2.medianBlur(image, 5)

    #image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]

correctedDiff = imageStack
plt.imshow(correctedDiff[0], cmap="gray")
plt.show()

initialAvg = np.nanmean(intensityImages[0])
initialStd = np.nanstd(intensityImages[0])
imageStack = []

for image in intensityImages:
    print("looping")
    avg = np.nanmean(image)
    std = np.nanstd(image)

    #image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image)) #renormalises - after contrast adjust so shows
    #image = image * 255

    newImage = initialAvg + ((image - avg) * (initialStd / std))

    np.clip(newImage, 0, 255)
    image = np.array(newImage, dtype = "uint8")
    image = cv2.medianBlur(image, 5)

    image = np.array(image, dtype="uint8")

        
    imageStack = imageStack + [image]
 
intensityImages = imageStack
 
correctedInt = imageStack
differenceImages = np.array(correctedDiff, dtype = "uint8")

refDiff = differenceImages[0]
refIntensity = intensityImages[0]
#print("pixel info")
#print(refDiff[110][110])

alignedIntensity = []

for i in range(1, len(differenceImages)):

    #intensityImg, h = alignImages(refIntensity, intensityImages[i])
    #alignedIntensity = alignedIntensity + [intensityImg]

    intensityImg, h = alignImages(refDiff, differenceImages[i])
    alignedIntensity = alignedIntensity + [intensityImg]

    # Print estimated homography
    #print("Estimated homography : \n",  h)
print(len(alignedIntensity))

plt.figure()
plt.axis("off")

ax1 = plt.subplot(1, 2, 1)
#ax2 = plt.subplot(1, 2, 2)

img1 = alignedIntensity[0]
#img2 = correctedIntensities[0]

fig1 = ax1.imshow(img1, cmap="gray", vmin = 100, vmax = 150)
#fig2 = ax2.imshow(img2, cmap="gray", vmin = 2, vmax = 25)

#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 9, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = alignedIntensity[sb.val]
    #img2 = correctedIntensities[sb.val]

    fig1.set_data(img1)
    #fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("Intensities")
#ax2.set_title("differences")


print("plotted")
sb.on_changed(update)
plt.show()
  





  