import cv2 as cv
import numpy as np
from math import log10, sqrt
import matplotlib.pyplot as plt

def PSNR(original, compressed):
    """
        Calculate the PSNR of the input image and a compressed/denoised image and see the effectiveness of denoising

        Arg:
            original: OG image input 
            compressed: image after processing
    """
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def filter_apply(filter_type,img,filter_size):
    """
        Apply filters on the input image based on need
        
        Args:
            filter_type: gauss,mean or median denoise filters
            img: input img
            filter_size: control parameter for denoising
    """
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
""" ________________________ PART 1 ________________________"""
def denoise(img,name,size=3,add_noise=False):
    """
        Display the results of denoising
    """    
    #Introduce Noise if needed:
    if add_noise==True:
        n_img = noise('gauss',img,100,1.5)
        title = ' Noise Added'
    else:
        n_img = img
        title=''

    # Find the required values
    f_img = filter_apply(name,n_img,size)
    psnr = PSNR(img,f_img)

    
    # Plotting
    fig, ax = plt.subplots(2,2,figsize=(20,10),gridspec_kw={'height_ratios': [5, 1]})
    ax[0,0].imshow(img,cmap='gray')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(f_img,cmap='gray')
    ax[0,1].set_title(f'Denoised Image using filter: {name+title}\nPSNR: {psnr}')
    ax[1,0].hist(img.flatten(),bins=255)
    ax[1,1].hist(f_img.flatten(),bins=255)

""" ________________________ PART 2 ________________________"""
def contrast_enchance(img,name,size=3,add_noise=False,cl=4):

    """
        Use CLAHE to do Contrast Enhancement on the image and plot the results
    """    
    clahe = cv.createCLAHE(clipLimit=cl,tileGridSize=(5,5))
    # Find the required values
    # Add optional Noise to the image
    if add_noise==True:
        n_img = noise('gauss',img,100,1.5)
        title = ' Noise Added'
    else:
        n_img = img
        title=''
    # Apply  Denoise filter on image
    f_img = filter_apply(name,n_img,size)
    ## Do CLAHE on each of the filtered images
    clahe_img = clahe.apply(f_img)


    # Plotting
    fig, ax = plt.subplots(2,2,figsize=(20,10),gridspec_kw={'height_ratios': [5, 1]})
    ax[0,0].imshow(img,cmap='gray')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(clahe_img,cmap='gray')
    ax[0,1].set_title(f'Constrast Enhanced Image\nDenoised using filter: {name+title}')
    ax[1,0].hist(img.flatten(),bins=255)
    ax[1,1].hist(clahe_img.flatten(),bins=255)


