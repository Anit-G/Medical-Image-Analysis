import numpy as np
def get_histogram(flat, bins):
    # create histogram function
    # zeros array with size of bins
    histogram = np.zeros(bins)
    # Flatten the image 
    #flat = img.flatten()
    # Scale flat
    flat = flat/(1+max(flat))*bins

    # loop through pixels and sum up counts of pixels
    for p_value in flat:
        histogram[int(p_value)] += 1
    
    # return our final result
    return histogram

def cummalative_sum(a):
    # Finds the CDF of histogram of the image
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1]+i)
    return np.array(b)

def histogram_equ(img):
    # Find histogram of image
    flat = img.flatten()
    hist = get_histogram(flat,bins=256)

    # Rescale the histogram using CDF of hist
    cs = cummalative_sum(hist)
    scale = (cs-cs.min())/(cs.max()-cs.min())*(256-1)
    img_new = scale[flat]
    
    img_new = np.reshape(img_new, img.shape)
    return img_new

