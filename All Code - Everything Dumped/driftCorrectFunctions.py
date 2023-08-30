import numpy as np
import h5py
import cv2
import time
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
import skimage
import dask.array as da
from skimage.filters import threshold_otsu
#import dask.dataframe as df
#import Registration

def calculateSobel(img):
        #sobel = Registration.crop_and_filter(self.original.rechunk({0: self.dE}), sigma=self.sigma,
        #                                     finalsize=self.fftsize * 2)
        #set dE=4 here - looked like that was what code did
        #I think need sigma=3 instead of imstack.sigma - code looked like it was giving each stack sigma=3 but not sure
        
        #couldn't get registration libary to work - will try and replace with scikit image crop and apply sobel filter seperately.
        #sobel = Registration.crop_and_filter(imStack.original.rechunk({0: 4}), sigma = 3, finalsize=imgStack.fftsize * 2)
        #imgStack = da.from_array(imgStack)
        #cropped = skimage.util.crop(imgStack, da.rechunk(imgStack, chunks = {0: 4}))
        #cropped = skimage.util.crop(da.rechunk(imgStack, chunks = {0: 4}), 3)
        sobel = skimage.filters.sobel(img)
        sobel = da.from_array(sobel)
        sobel = da.rechunk(sobel, chunks = {0: 4})
        #sobel = skimage.filters.sobel(cropped)
        #newSobel = (sobel - sobel.mean(axis=(1, 2), keepdims=True))
        newSobel = (sobel - np.mean(sobel))

        return newSobel

def crossCorrelations(sobelImgStack):
      #I think this should score similarity in edge positions maybe

      #crossCorrelations = dask.cross_corr(sobelImgStack)
      #crossCorrelations = df.DataFrame.corr(sobelImgStack)

      crossCorrelations = []

      for img in sobelImgStack:

        crossCorrelation = skimage.registration.phase_cross_correlation(np.array(sobelImgStack[0]), np.array(img))
        crossCorrelations = crossCorrelations + [crossCorrelation[1]]

      print("cross correlations")
      print(crossCorrelations)

      return crossCorrelations

def findMaxAndArg(crossCorrelations):
     max = np.max(crossCorrelations)
     argMax = np.argmax(crossCorrelations)

     return max, argMax

def thresholding(sobelImgStack): 
    thresh = threshold_otsu(da.from_array(sobelImgStack[0]))
    plt.plot(thresh)
    plt.show()




#left out for now
#don't really understand what this should do
#only used for values in thresholding later - gunna see if can use something else
#def findHalfMatrix(max, argmax):
#     #self.W, self.DX_DY = Registration.calculate_halfmatrices(self.weights, self.argmax, fftsize=self.fftsize)
#     w, dx_dy = skimage.registration.calculate_halfmatrices(max, argmax, 512)
#     return w, dx_dy
        


# open the file as 'f'
with h5py.File('Diamond images\medipix-321127.hdf', 'r') as f:

    #only key is entry, but this looks for it
    key = list(f.keys())[0]
    data = np.array(f[key]["instrument"]["detector"]["data"])

    imgStack = data[0][:] #stack of 20 images at 1st energy
    sobelImgStack = []

    #find edges with Sobel operator
    for img in imgStack:
        sobelImg = calculateSobel(img)
        sobelImgStack = sobelImgStack + [sobelImg]

    #for array of correlation scores - use  to find match
    correlation = crossCorrelations(sobelImgStack)
    max, argmax = findMaxAndArg(crossCorrelations)
    #print("correlation = " + str(correlation))
    #to show first image in stack
    #findHalfMatrix(max, argmax)
    thresholding(sobelImgStack)
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    ax1.imshow(sobelImgStack[0], cmap="gray", vmin=-0.00000000001, vmax = 0.00000000001)
    ax2.imshow(sobelImgStack[1], cmap="gray", vmin=-0.00000000001, vmax = 0.00000000001)
    ax3.imshow(imgStack[0], cmap = "gray", vmin = 300, vmax = 700)
    ax4.imshow(imgStack[1], cmap = "gray", vmin = 300, vmax = 700)

    plt.show()