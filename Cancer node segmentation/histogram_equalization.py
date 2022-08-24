import numpy as np
import cv2 as cv
from math import log10,sqrt
import pandas as pd

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

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def preprocess(img,test_case='mg',sharpen='none',ce=True):
    """
        Take in a DICOM file format image and preprocess it with median+gaussian
        + CLAHE then out the result as a numpy array of single dimension
    """
    #Convert to numpy array
    img = np.asarray(img.pixel_array,dtype=np.int)

    # standardize the img
    img = img/img.max()*255
    img_og = np.asarray(img,dtype=np.float32)

    # Filtering
    if test_case=='mg':
        img = filter_apply('median',img_og,3)
        img = filter_apply('gauss',img,3)
    elif test_case=='m':
        img = filter_apply('median',img_og,3)
    elif test_case=='g':
        img = filter_apply('gauss',img_og,3)
    elif test_case =='nl':
        img = cv.fastNlMeansDenoising(img_og)
    
    psnr = PSNR(img_og,img)

    # Image Sharpening
    if sharpen=='ghpf':
        blur = cv.GaussianBlur(img,(3,3),50)
        img = cv.subtract(img,blur)
        #img = cv.add(img,127*np.ones(img.shape))
    elif sharpen =='conv':
        kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
        img = cv.filter2D(src=img, ddepth=-1, kernel=kernel3)
    
    if ce==True:
        # CLAHE
        img = img.astype(np.uint8)
        clahe = clahe = cv.createCLAHE(clipLimit=4,tileGridSize=(5,5))
        img = clahe.apply(img)
        img = filter_apply('gauss',img,5)
    
    return img,psnr


import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

def watersheding(img,thresh):

    # Compute Euclidean distance from every binary pixel
    # to the nearest zero pixel then find peaks
    distance_map = ndimage.distance_transform_edt(thresh)
    local_max = peak_local_max(distance_map, indices=False, min_distance=20, labels=thresh)

    # Perform connected component analysis then apply Watershed
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=thresh)

    # Iterate through unique labels
    total_area = []
    contours = []
    for label in np.unique(labels):
        if label == 0:
            continue

        # Create a mask
        mask = np.zeros(img.shape, dtype="uint8")
        blank = np.zeros(img.shape, dtype="uint8")
        mask[labels == label] = 255

        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        total_area.append(area)

        cv2.drawContours(blank, [c], -1, (36,255,12), -1)
        contours.append(blank)
    
    return total_area,contours,labels



from sklearn.metrics import jaccard_score,f1_score,accuracy_score
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
def validation(annot,morph,water,idx = ['Active Contours','Watersheding']):
    annot= annot/255
    if idx[1] == 'Watersheding':
        water = water/36

    annot = annot.astype('uint8')
    water = water.astype('uint8')
    
    jm = jaccard_score(annot,morph,average='micro')
    jw = jaccard_score(annot,water,average='micro')

    fm = f1_score(annot,morph,average='micro')
    fw = f1_score(annot,water,average='micro')

    accm = accuracy_score(annot,morph)
    accw = accuracy_score(annot,water)

    hm = directed_hausdorff(annot,morph)
    hw = directed_hausdorff(annot,water)

    d = {'Jaccard Score':[jm,jw],'Accuracy Score':[accm,accw],'F1 Score':[fm,fw],'Hausdorff Distance':[hm[0],hw[0]]}
    
    return pd.DataFrame(d,index=idx)

"""Alpha Beta Swap and Expansion"""


from maxflow import fastmin as mp
import matplotlib.pyplot as plt

def alpha_seg(img,img_seg):
    labels = img_seg
    D = np.zeros(labels.shape+(2,),dtype=float)
    D[:,:,0] = labels**2 # beta label
    D[:,:,1] = (labels-1)**2 # alpha label

    V = np.array([[0,1],[1,0]],dtype=float)
    optimized_label_exp = mp.aexpansion_grid(D,3*V,labels=labels.copy())
    optimized_label_swap = mp.abswap_grid(D,3*V,labels=labels.copy())

    fig, ax = plt.subplots(2,2,figsize=(10,10))

    ax[0,0].imshow(labels,cmap='gray')
    ax[0,0].set_title('Original Segmentaion')
    ax[0,1].imshow(optimized_label_exp,cmap='gray')
    ax[0,1].set_title('Alpha Expansion')
    ax[1,1].imshow(optimized_label_swap,cmap='gray')
    ax[1,1].set_title('Alpha Beta Swap')
    ax[1,0].imshow(img,cmap='gray')
    ax[1,0].set_title('Original Image')
    plt.show()

    return optimized_label_swap,optimized_label_exp
