import torch
#from DeepReg.deepreg.model import DeepRegModel
from DeepReg.deepreg.dataset import pretrained
import numpy as np
import h5py
import cv2
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.widgets import Slider
from skimage import io
from skimage.io import imsave as sk_imsave

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


def rotateImg(img, da):

    h, w = img.shape
    center = (w / 2, h / 2)

    scale = 1
    m = cv2.getRotationMatrix2D(center, da, scale)
    rotatedImg = cv2.warpAffine(img, m, (w,h))
    return rotatedImg

def align_images_deepreg(image1, image2):
    # Load pre-trained DeepReg model
    model = DeepRegModel.load_model("path_to_pretrained_model")

    # Convert images to grayscale and normalize
    gray_image1 = image1 / 255.0
    gray_image2 = image2 / 255.0

    # Prepare input tensor (add batch dimension)
    input_tensor = torch.from_numpy(np.stack([gray_image1, gray_image2], axis=0)).unsqueeze(0)

    # Move the input tensor to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # Run the forward pass through the DeepReg model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Extract the estimated transformation from the output tensor
    estimated_transform = output_tensor[0, ...].cpu().numpy()

    # Warp the input image using the estimated transformation
    rows, cols, _ = image2.shape
    warped_image = cv2.warpAffine(image1, estimated_transform[:2], (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return warped_image




imageNumbers = ["321124-321125", "321127-321136", "321227-321236", "321457-321466", "321527-321536", "321617-321626", "321627-321636"]
differenceImages = []
intensityImages = []

for i in imageNumbers: 

    differenceImg = io.imread(r"RC294 - Cross Centre\%s_Diffference.tif" % str(i))
    intensityImg = io.imread(r"RC294 - Cross Centre\%s_Intensity.tif" % str(i))

    differenceImages = differenceImages  + [differenceImg]
    intensityImages = intensityImages + [intensityImg]

    

initialIntensityImages = intensityImages
initialDifferenceImages = differenceImages

imageStack = []

initialAvg = np.nanmean(differenceImages[0])
initialStd = np.nanstd(differenceImages[0])

for image in differenceImages:
    avg = np.nanmean(image)
    std = np.nanstd(image)

    
    newImage = initialAvg + ((image - avg) * (initialStd / std))
    #print(newImage)
    np.clip(newImage, -1, 1)
    
    image = newImage
    image = (image + 1)/2
    image = image * 255
    image = np.array(image, dtype="uint8")
    image = cv2.medianBlur(image, 5)
    image = np.clip(image, 120, 140)
    image = (image-np.nanmin(image))/(np.nanmax(image)-np.nanmin(image))
    image = image * 255
    image = np.array(image, dtype="uint8")

    #plt.imshow(image, cmap="gray")
    #plt.show()

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

angles = [0, 0, -30, -45, -50, -50, -90]

for i in range(1, len(intensityImages)):
    print("before rotate")
    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()

    intensityImages[i] = rotateImg(intensityImages[i], angles[i])
    differenceImages[i] = rotateImg(differenceImages[i], angles[i])

    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()


firstImg = differenceImages[0]
for i in range(1, len(differenceImages)):
    img = differenceImages[i]
    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()
    differenceImages[i] = align_images_deepreg(img, firstImg)

    #plt.imshow(differenceImages[i], cmap="gray")
    #plt.show()

correctedDifferences = differenceImages
correctedIntensities = intensityImages

plt.figure()
plt.axis("off")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

img1 = correctedDifferences[0]
img2 = correctedIntensities[0]

fig1 = ax1.imshow(img1, cmap="gray")
fig2 = ax2.imshow(img2, cmap="gray")


#defines slider axis
axs = plt.axes([0.15, 0.001, 0.65, 0.03])
sb = Slider(axs, 'image no', 0, 6, valinit=0, valstep = 1)

#runs when slider moved
def update(val):
    img1 = correctedDifferences[sb.val]
    img2 = correctedIntensities[sb.val]

    fig1.set_data(img1)
    fig2.set_data(img2)
        
    plt.draw()

ax1.set_title("differences")
ax2.set_title("intensities")

sb.on_changed(update)

averageIntensity = averageImages(intensityImages)
averageDifference = averageImages(differenceImages)

ax4.imshow(averageIntensity, cmap="gray")
ax3.imshow(averageDifference, cmap = "gray")

#for i in range(0, len(correctedDifferences)):
    #sk_imsave("rotatedCross/%s_D.tif" % (str(i)), correctedDifferences[i])
    #sk_imsave("rotatedCross/%s_I.tif" % (str(i)), correctedIntensities[i])

plt.show()


