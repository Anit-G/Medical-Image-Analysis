import cv2 as cv
import numpy as np
from math import log10, sqrt
import matplotlib.pyplot as plt
from Helper import filter_apply, noise
from hough_transform import hough_transform

def convolution(img,kernel):
    assert len(img.shape) == 2

    #Convolution of the image
    conv_img = cv.filter2D(img,-1,kernel)
    
    return conv_img*10

def sobel(img,ksize):
    """ Use sobel Edge detection to the edge in an image"""
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #Gaussian Blur
    Gauss_blur_img = cv.GaussianBlur(img,ksize,0,0)

    # X sobel filter
    new_image_x = cv.filter2D(Gauss_blur_img,-1,filter)
    # Y sobel Filter
    new_image_y = cv.filter2D(Gauss_blur_img,-1,filter)

    # Sobel = sqrt(x^2+y^2)
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_magnitude = gradient_magnitude.astype('uint8')
    # Edge detection
    #s_img = cv.Sobel(src=Gauss_blur_img, ddepth=cv.CV_32F, dx=1, dy=1, ksize=5)
    return gradient_magnitude/10

def laplacian(img,ksize):
    """ Use laplacian Edge detection to the edge in an image"""
    #Gaussian Blur
    Gauss_blur_img = cv.GaussianBlur(img,(3,3),0,0)
    # Edge detection
    l_img = cv.Laplacian(src=Gauss_blur_img, ddepth=cv.CV_64F, ksize=5)
    return l_img

def canny_edge_detection(img,ksize,maxval,minval):
    
    #Gaussian Blur
    Gauss_blur_img = cv.GaussianBlur(img,ksize,0,0)
    
    #Canny edge detection
    canny_edge = cv.Canny(Gauss_blur_img,minval,maxval,ksize)
    return canny_edge

def contrast_enchance(img,name,size=3,add_noise=False):

    """
        Use CLAHE to do Contrast Enhancement on the image and plot the results
    """    
    clahe = cv.createCLAHE(clipLimit=3,tileGridSize=(5,5))
    # Find the required values
    # Add optional Noise to the image
    if add_noise==True:
        n_img = noise('gauss',img,100,1.5)
    else:
        n_img = img
    # Apply  Denoise filter on image
    f_img = filter_apply(name,n_img,size)
    ## Do CLAHE on each of the filtered images
    clahe_img = clahe.apply(f_img)

    return clahe_img

def edge_detection(img,img_name,kernel,ksize=(3,3),maxval=100,minval=70):
    """
        Use multiple types of edge detection methods to find the edge of the sternum by defining the appropreiate ROI
        Methods: Sobel, Canny edge detection and convolution of a kernel
    """
    # Image Cropping
    if img_name=='img1':
        img = img[190:350,300:650]
    elif img_name=='img2':
        img = img[30:120,370:550]
    else:
        print('Invalid img_name')

    # Contrast Enhancment
    img = contrast_enchance(img,'median')

    conv_img = convolution(img,kernel)
    sobel_img = sobel(img,(9,9))
    laplace_img = laplacian(img,ksize)
    canny_img = canny_edge_detection(img,ksize=ksize,maxval=maxval,minval=minval)

    # Plotting the images
    fig,ax = plt.subplots(2,3,figsize=(20,10))
    #Images
    ax[0,0].imshow(laplace_img,cmap='gray')
    ax[0,0].set_title('Edge Detect using Laplacian')
    ax[0,1].imshow(conv_img,cmap='gray')
    ax[0,1].set_title('Edge Detect using Convolution')
    ax[0,2].imshow(img,cmap='gray')
    ax[0,2].set_title('Original Image')
    #Images
    ax[1,0].imshow(sobel_img,cmap='gray')
    ax[1,0].set_title('Sobel Edge Detection')
    ax[1,1].imshow(canny_img,cmap='gray')
    ax[1,1].set_title('Canny Edge Detection')
    ax[1,2].imshow(img,cmap='gray')
    ax[1,2].set_title('Original Image')

    
    return canny_img


def lame_detect(img,img_name,ksize,maxval,minval,x0,y0,threshold=100,num_theta=1000,n=1.5):
    # Image Cropping
    if img_name=='img1':
        img = img[190:350,300:650]
    elif img_name=='img2':
        img = img[30:120,370:550]
    else:
        print('Invalid img_name')

    # Contrast Enhancment
    img = contrast_enchance(img,'median')
    # Canny edge detection
    canny_img = canny_edge_detection(img,ksize=ksize,maxval=maxval,minval=minval)
    para = hough_transform(canny_img,threshold,num_theta,n)

    plt.figure()
    plt.imshow(canny_img)
    t = np.linspace(0, 2 * np.pi, 100)
    for a0,b0 in para:
        x = ((np.abs(np.cos(t))) ** (2 / n)) * b0 * np.sign(np.cos(t))+x0
        y = ((np.abs(np.sin(t))) ** (2 / n)) * a0 * np.sign(np.sin(t))+y0
        plt.plot(x,y)
    plt.show()
