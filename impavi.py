"""
Module of useful functions for Image Processing and Artificial Vision

@author: Ignacio Cordova Pou
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import skimage
from scipy.ndimage.measurements import histogram
from skimage.filters.rank import entropy
from sklearn.cluster import KMeans
from scipy import ndimage


def luminance(rgb):
    """
    Calculates the luminance of an RGB image

    Input: 3-channel RGB image
    Output: Luminance image
    """
    l_im =  0.299*np.double(rgb[:,:,0])+\
            0.587*np.double(rgb[:,:,1])+\
            0.114*np.double(rgb[:,:,2])
    

    return l_im 



def equalize(im,plot=False):
    
    """"
    Equalize the histogram of a one-channel image.

    Inputs: gray-scale image and plot flag
    Output: equalized image
    """
    
    h = histogram(im,0,255,256)
    hc = np.cumsum(h)/(im.shape[0]*im.shape[1])
    
    im_eq = hc[im]
   
    h_im_eq = histogram(im_eq,0,1,256)
    h_im_eq_c = np.cumsum(h_im_eq)/(im.shape[0]*im.shape[1])

    if plot:

        plt.figure(constrained_layout=True,figsize=(20,20))

        plt.subplot(321)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')


        plt.subplot(322)    
        plt.plot(h)
        plt.title('Original Histogram')

        plt.subplot(323)    
        plt.plot(255*hc)
        plt.title('Histogram Cumulative Sum Normalized of Original Image')

        plt.subplot(326)    
        plt.plot(h_im_eq)
        plt.title('Histogram of Equalized Image')

        plt.subplot(325)
        plt.imshow(im_eq,cmap='gray')
        plt.title('Equalized')
        plt.axis('off')


        plt.subplot(324)    
        plt.plot(255*h_im_eq_c)
        plt.title('Histogram Cumulative sum of Equalized Image')
        
    return im_eq


def high_pass_filter(im,radius,plot=False):
    """
    Applies a high pass filter to an image.

    Inputs: gray-scale image and radius of the filter and plot flag
    Output: high pass filtered image

    """
    im_tf = np.fft.fftshift(np.fft.fft2(im))
    #build Laplacian Filter 
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    lf = u**2 + v**2
    #sharp cut-off
    circ = lf>radius

    im1 = np.abs(np.fft.ifft2(im_tf*circ))

    if plot:
        plt.figure(constrained_layout=True,figsize=(20,20))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(im1,cmap='gray')
        plt.title('Laplacian Filter')
        plt.axis('off')

    return im1

def low_pass_filter(im,radius,plot=False):
    """
    Applies a low pass filter to an image.
    
    Inputs: gray-scale image and radius of the filter and plot flag
    Output: low pass filtered image
    """
    im_tf = np.fft.fftshift(np.fft.fft2(im))
    #build Laplacian Filter 
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    lf = u**2 + v**2
    #sharp cut-off
    circ = lf<radius

    im1 = np.abs(np.fft.ifft2(im_tf*circ))

    if plot:
        plt.figure(constrained_layout=True,figsize=(7,7))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(im1,cmap='gray')
        plt.title('Laplacian Filter')
        plt.axis('off')

    return im1


def median_filters(im,plot=False):
    """
    Applies a series of median filters to an image.

    Inputs: gray-scale image and plot flag
    Output: median filtered image
    """

    media1 = scipy.signal.medfilt2d(im, kernel_size=[11,11])
    media2 = scipy.signal.medfilt2d(media1, kernel_size=[41,41])
    media3 = scipy.signal.medfilt2d(media2, kernel_size=[21,21])
    media4 = scipy.signal.medfilt2d(media3, kernel_size=[21,21])
    if plot:
        plt.figure(constrained_layout=True,figsize=(20,20))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(media2,cmap='gray')
        plt.title('Double Median Filter')
        plt.axis('off')
    return media2


def FS(ast):
       
    """
    Floyd-Steinbeirg (FS) algorithm for error compensation 

    Input: ast- 2D array (gray scale) normalized to 1
    Output: Binarized image 
    """
    
    
    for i in range(1,511): #skips the edges 
        for j in range(1,511):
            
            #error 
            px = ast[i,j]
            if px>0.5:
                error = px-1

            else:
                error = px
            
            ast[i,j+1]=ast[i,j+1]     + (7./16.)*error
            ast[i+1,j+1]=ast[i+1,j+1] + (1./16.)*error
            ast[i+1,j]=ast[i+1,j]     + (5./16.)*error
            ast[i+1,j-1]=ast[i+1,j-1] + (3./16.)*error
                
                
    imFS = ast>0.5
    return imFS 


def otsu_filter(im,plot=False):
    """
    Applies an Otsu threshold to an image.
    
    Inputs: gray-scale image and plot flag
    Output: Otsu filtered image
    """
    otsu = im>skimage.filters.threshold_otsu(im)
    if plot:
        plt.figure(constrained_layout=True,figsize=(7,7))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(otsu,cmap='gray')
        plt.title('Otsu Filter')
        plt.axis('off')
    return otsu



def JJN(ast):
    """
    Jarvis-Judice-Ninke (JJN) algorithm for error compensation
    
    Input: ast- 2D array (gray scale) normalized to 1
    Output: Binarized image
    """

    for ii in range(1,510): #skips the edges
        for jj in range(1,510):
            
            #error
            px = ast[ii,jj]
            if px>0.5:
                error = px-1
                
            else:
                error = px
            
            error = error /48 
            ast[ii,jj+1] += 7.*error
            ast[ii,jj+2] += 5.*error
            ast[ii+1,jj-2] +=3.*error 
            ast[ii+1,jj-1] +=5.*error 
            ast[ii+1,jj-0] +=7.*error 
            ast[ii+1,jj-0] +=5.*error 
            ast[ii+1,jj+1] +=3.*error 
            ast[ii+2,jj-2] +=1.*error 
            ast[ii+2,jj-1] +=3.*error 
            ast[ii+2,jj-0] +=5.*error 
            ast[ii+2,jj-0] +=3.*error 
            ast[ii+2,jj+1] +=1.*error
    
    
    imJJN = ast>0.5
    return imJJN 



def kirsch_compass_kernel(im):
    """
    Applies a Kirsch compass filter to an image.
    
    Inputs: gray-scale image
    Output: Kirsch compass filtered image
    """

    #Es una matriz compuesta de 8 matrices 3x3
    kir = np.zeros([8, 3, 3])
    g1 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    g2 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    g3 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    g4 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    g5 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    g6 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    g7 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    g8 = np.array([[-3, 5, 5],[-3,0,5],[-3,-3,-3]])
    
    convo_kirsch=np.zeros((8, im.shape[0], im.shape[1]))

    convo_kirsch[0,:,:] = np.abs(ndimage.convolve(im, g1))
    convo_kirsch[1,:,:] = np.abs(ndimage.convolve(im, g2))
    convo_kirsch[2,:,:] = np.abs(ndimage.convolve(im, g3))
    convo_kirsch[3,:,:] = np.abs(ndimage.convolve(im, g4))
    convo_kirsch[4,:,:] = np.abs(ndimage.convolve(im, g5))
    convo_kirsch[5,:,:] = np.abs(ndimage.convolve(im, g6))
    convo_kirsch[6,:,:] = np.abs(ndimage.convolve(im, g7))
    convo_kirsch[7,:,:] = np.abs(ndimage.convolve(im, g8))

    im_kirsch = np.zeros((im.shape[0],im.shape[1]))

    for ii in range(im.shape[0]):
        for jj in range(im.shape[1]):
            im_kirsch[ii, jj] = np.amax(convo_kirsch[0:8, ii, jj])
                    
    return im_kirsch


def color_reduction(im):
    """
    Performs color dithering followinf the HSV color model.
    RGB images have 256x256x256 possible color combinations
    This function reduces the colors to 6x6x6 possible combinations

    Inputs: RGB image
    Output: Dithered image"""
    
    
    #Build the look up table 
    array = np.arange(0,256,1)
    a = np.rint(array//43)*43
    
    
    #apply to RGB image 
    im6x6x6 = a[im]
    
    #Output is float so we MUST normalize! 
    im6x6x6 = im6x6x6/im6x6x6.max()
    
    return im6x6x6


def apply_kmeans(im,k,plot=False):
    """
    Applies k-means clustering to an image and automatically selects the largest cluster.
    
    Inputs: gray-scale image and number of clusters and plot flag
    Output: largest cluster
    """
    dataK = im.reshape(im.shape[0]*im.shape[1],1)
    kmn = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(dataK)
    # a label (0 to k) is assigned to each sample (row)
    labels = kmn.predict(dataK)

    # from 1d-array to 2d-array
    imRes = np.reshape(labels, [im.shape[0], im.shape[1]])

    ref = 0 

    for label in range(k):
        cluster_size = np.count_nonzero(imRes==label)
        if cluster_size>ref:
            ref = cluster_size
            final_res = imRes==label
    
    if plot:
        plt.figure(constrained_layout=True,figsize=(11,11))
        plt.subplot(121)
        plt.imshow(im,cmap='gray')
        plt.title('Input')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imRes,cmap='gray')
        plt.title('Largest K-means cluster')
        plt.axis('off')
    
    return 1-final_res


"""
The following functions were specially built for a project that 
you can find in my github.
"""

def show(string):
    """
    Shows an image saved as string

    Inputs: string
    Output: None
    """
    im = plt.imread(string)
    plt.figure(constrained_layout=True,figsize=(8,8))
    plt.imshow(im,cmap='gray')
    plt.axis('off')

def hfreq_segmentation(string):
    """
    Applies a high-frequency based segmentation to an image. 
    
    Inputs: string
    Output: segmented image
    """
    im = plt.imread(string)
    im_eq = equalize(im,plot=False)
    im_eq_laplacian = high_pass_filter(im_eq,radius=0.005,plot=False)
    median = median_filters(im_eq_laplacian,plot=False)
    im_seg = apply_kmeans(median,7,plot=False)
    
    return im_seg

def edge_enhanced_segmentation(string):
    """
    Applies a edge-enhanced segmentation to an image.
    
    Inputs: string
    Output: segmented image
    """
    im = plt.imread(string)
    im_eq = equalize(im,plot=False)
    im_eq_kirsch = kirsch_compass_kernel(im_eq)
    median = median_filters(im_eq_kirsch,plot=False)
    im_seg = apply_kmeans(median,2,plot=False)
    
    return im_seg


def get_ssim(string,mode=0):
    """
    Computes the structural similarity index between the results obtained and the manually segmented image.
    mode = 0 performs the segmentation using the high-frequency based segmentation
    mode = 1 performs the segmentation using the edge-enhanced segmentation

    Inputs: string and mode
    Output: SSIM
    """
    im_manual = plt.imread(string+'_manual.png')
    im_topman = plt.imread(string+'_topman.png')
    im_tscratch = plt.imread(string+'_tscratch.png')
    im_multiceg_Seg = plt.imread(string+'_multiCellSeg.png')

    if mode == 0:
        my_segmentation = hfreq_segmentation(string+'.tif')
    if mode == 1: 
        my_segmentation = edge_enhanced_segmentation(string+'.tif')

    ssim_topman = skimage.metrics.structural_similarity(im_manual,im_topman)
    ssim_tscratch = skimage.metrics.structural_similarity(im_manual,im_tscratch)
    ssim_multiceg_Seg = skimage.metrics.structural_similarity(im_manual,im_multiceg_Seg)
    ssim_mine = skimage.metrics.structural_similarity(im_manual,my_segmentation)


    return [ssim_topman,ssim_tscratch,ssim_multiceg_Seg,ssim_mine]

def state_of_the_art(string):
    """
    Computes the structural similarity index between the results obtained and the manually segmented image.
    Plots the each one of the results with its respective SSIM.
    
    Inputs: string
    Output: none
    """
    im_manual = plt.imread(string+'_manual.png')
    im_topman = plt.imread(string+'_topman.png')
    im_tscratch = plt.imread(string+'_tscratch.png')
    im_multiceg_Seg = plt.imread(string+'_multiCellSeg.png')

    ssim_topman = skimage.metrics.structural_similarity(im_manual,im_topman)
    ssim_tscratch = skimage.metrics.structural_similarity(im_manual,im_tscratch)
    ssim_multiceg_Seg = skimage.metrics.structural_similarity(im_manual,im_multiceg_Seg)
    plt.figure(figsize=(20,20))
    plt.subplot(141)
    plt.imshow(im_manual,cmap='gray')
    plt.title('Manually segmented (target)')

    plt.subplot(142)
    plt.imshow(im_topman,cmap='gray')
    plt.title('Topman SSIM={:.2f}'.format(ssim_topman))

    plt.subplot(143)
    plt.imshow(im_tscratch,cmap='gray')
    plt.title('Tscratch SSIM={:.2f}'.format(ssim_tscratch))

    plt.subplot(144)
    plt.imshow(im_multiceg_Seg,cmap='gray')
    plt.title('MultiCellSeg SSIM={:.2f}'.format(ssim_multiceg_Seg))



def ssim_matrix(string,final):
    """
    Computes the structural similarity index between the results obtained and the manually segmented image.
    It also computes the SSIM between the results obtained.
    Builds a correlation matrix using all SSIMs
    
    Inputs: string and my final result of segmentation
    Output: correlation matrix
    """
    im_manual = plt.imread(string+'_manual.png')
    im_topman = plt.imread(string+'_topman.png')
    im_tscratch = plt.imread(string+'_tscratch.png')
    im_multiceg_Seg = plt.imread(string+'_multiCellSeg.png')

    #build a confussion matrix with all ssim values
    ssim = np.identity(5)
    ssim12 = skimage.metrics.structural_similarity(im_manual,im_topman)
    ssim13 = skimage.metrics.structural_similarity(im_manual,im_tscratch)
    ssim14 = skimage.metrics.structural_similarity(im_manual,im_multiceg_Seg)
    ssim15 = skimage.metrics.structural_similarity(im_manual,final)
    ssim[0,1] = np.round(ssim12,3)
    ssim[0,2] = np.round(ssim13,3)
    ssim[0,3] = np.round(ssim14,3)
    ssim[0,4] = np.round(ssim15,3)
    ssim21 = skimage.metrics.structural_similarity(im_topman,im_manual)
    ssim22 = skimage.metrics.structural_similarity(im_topman,im_tscratch)
    ssim23 = skimage.metrics.structural_similarity(im_topman,im_multiceg_Seg)
    ssim24 = skimage.metrics.structural_similarity(im_topman,final)
    ssim[1,0] = np.round(ssim21,3)
    ssim[1,2] = np.round(ssim22,3)
    ssim[1,3] = np.round(ssim23,3)
    ssim[1,4] = np.round(ssim24,3)
    ssim31 = skimage.metrics.structural_similarity(im_tscratch,im_manual)
    ssim32 = skimage.metrics.structural_similarity(im_tscratch,im_topman)
    ssim33 = skimage.metrics.structural_similarity(im_tscratch,im_multiceg_Seg)
    ssim34 = skimage.metrics.structural_similarity(im_tscratch,final)
    ssim[2,0] = np.round(ssim31,3)
    ssim[2,1] = np.round(ssim32,3)
    ssim[2,3] = np.round(ssim33,3)
    ssim[2,4] = np.round(ssim34,3)
    ssim41 = skimage.metrics.structural_similarity(im_multiceg_Seg,im_manual)
    ssim42 = skimage.metrics.structural_similarity(im_multiceg_Seg,im_topman)
    ssim43 = skimage.metrics.structural_similarity(im_multiceg_Seg,im_tscratch)
    ssim44 = skimage.metrics.structural_similarity(im_multiceg_Seg,final)
    ssim[3,0] = np.round(ssim41,3)
    ssim[3,1] = np.round(ssim42,3)
    ssim[3,2] = np.round(ssim43,3)
    ssim[3,4] = np.round(ssim44,3)

    ssim[4,0] = np.round(ssim15,3)
    ssim[4,1] = np.round(ssim24,3)
    ssim[4,2] = np.round(ssim34,3)
    ssim[4,3] = np.round(ssim44,3)
    return ssim




