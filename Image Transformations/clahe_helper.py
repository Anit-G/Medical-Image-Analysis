import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt
import cv2 as cv

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# Add noise function
def noise(noise_type,img,var,amount):
    assert len(img.shape) == 2

    if noise_type == 's&p': 
        # Salt and Pepper noise
        s_vs_p = 0.5
        amount = 0.05
        out = np.copy(img)
    
        # Add salt
        num_salt = np.ceil(amount*img.size*s_vs_p)
        coords = [np.random.randint(0,i-1,int(num_salt)) for i in img.shape]
        out[coords] = 1

        # Add pepper
        num_pepper = np.ceil(amount*img.size*s_vs_p)
        coords = [np.random.randint(0,i-1,int(num_pepper)) for i in img.shape]
        out[coords] = 0

        return out

    if noise_type == 'gauss':
        out_g = np.copy(img)
        mean = 0        

        # Generate gaussian Noise
        # Using a normal distribution generate a random noise image
        gauss = cv.randn(out_g,mean,var**0.5)
        #gauss = np.random.normal(mean,sigma,(row,col))
        #gauss = gauss.reshape(row,col)
        
        # Add noise to image
        noise = gauss+img
        return noise
# Applying Denoising Filters on the image
# Mean Filter
# gaussian Filter
# Median Filter
def filter_apply(filter_type,img,filter_size):
    assert len(img.shape) == 2

    if filter_type == 'mean':
        return cv.blur(img,(filter_size,filter_size))
    elif filter_type == 'gauss':
        return cv.GaussianBlur(img,(filter_size,filter_size),sigmaX=50)
    elif filter_type == 'median':
        return cv.medianBlur(img,filter_size)
    else: 
        print('filter_type Invalid')
        return None

# Contrast Enhancment & Smoothening using CLAHE Method

def clahe(img,var,amount):
    # Apply noise to image
    sp_noise = noise('s&p',img,var,amount)
    gauss_noise = noise('gauss',img,var,amount)
    
    metrics = []
    for image,noise_type in zip([sp_noise,gauss_noise],['s&p','gauss']):
        clahe = cv.createCLAHE(clipLimit=4,tileGridSize=(5,5))
        eval_psnr = []

        for filter in ['mean','median','gauss']:
            # Apply  Denoise filter on image
            denoise_img = filter_apply(filter,image,3)
            ## Do CLAHE on each of the filtered images
            clahe_img = clahe.apply(denoise_img)
            ### Evaluate the performance using PSNR
            PSNR_value = PSNR(img,denoise_img)
            eval_psnr.append(PSNR_value)
            
        metrics.append(eval_psnr)
    
    return metrics

def clahe_plot(img,var,amount,name):
    # Apply noise to image
    sp_noise = noise('s&p',img,var,amount)
    gauss_noise = noise('gauss',img,var,amount)
        
    """---------------Salt and Pepper Noise----------------------"""

    # Apply  Denoise filter on image
    mean_img = filter_apply('mean',sp_noise,3)
    median_img = filter_apply('median',sp_noise,3)
    gauss_img = filter_apply('gauss',sp_noise,3)

    clahe = cv.createCLAHE(clipLimit=40,tileGridSize=(5,5))
    ## Do CLAHE on each of the filtered images
    mean_img1 = clahe.apply(mean_img)
    median_img1 = clahe.apply(median_img)
    gauss_img1 = clahe.apply(gauss_img)

    """------------------Gaussian Noise----------------"""
    # Apply  Denoise filter on image
    mean_img = filter_apply('mean',gauss_noise,3)
    median_img = filter_apply('median',gauss_noise,3)
    gauss_img = filter_apply('gauss',gauss_noise,3)

    ## Do CLAHE on each of the filtered images
    mean_img2 = clahe.apply(mean_img)
    median_img2 = clahe.apply(median_img)
    gauss_img2 = clahe.apply(gauss_img)

    fig, ax = plt.subplots(9,2,figsize=(20,40))
    Image = [img,sp_noise,mean_img1,median_img1,gauss_img1,gauss_noise,mean_img2,median_img2,gauss_img2]
    Title = ['Original','Salt and Peper Noise Image',
    'Filter: Mean, Noise: s%p',
    'Filter: Median, Noise: s%p',
    'Filter: Gaussian, Noise: s%p',
    'Gaussian Noise Image',
    'Filter: Mean, Noise: gauss',
    'Filter: Median, Noise: gauss',
    'Filter: Gaussian, Noise: gauss']
    for img,title,i in zip(Image,Title,range(9)):
        ax[i,0].imshow(img,cmap='gray')
        ax[i,0].axis('off')
        ax[i,0].set_title(title)
        ax[i,1].hist(img.flatten(),255)

    #fig.savefig(f'CLAHE_{name}.jpg')
        