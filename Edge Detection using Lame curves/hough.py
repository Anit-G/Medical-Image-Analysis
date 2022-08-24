from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import math

"""
    Input:
        num_theta
        num_rho
        img
        threshold
"""
# Range of Theta is 0-180 deg 
    # Increment = 180/num_theta
# Range of rho is 0-d (d is diagonal of image)
    # Increment = d/num_rho
def shortest_distance(x1, y1, m,b):
    """
        Perpendicular distance between a line and a point
    """
    d = abs((m*x1 - y1 + b)) / (math.sqrt(m*m + 1))
    return d


"""__________________ Hough Space ___________________"""
img = plt.imread('pentagon.png')
num_theta = 180
num_rho = 500
threshold = 100

coord = []

x_max = img.shape[0]
y_max = img.shape[1]

theta_min = 0.0
theta_max = 1.0*math.pi # get the pi value as a float
theta_arr = np.arange(0,theta_max+theta_max/num_theta,theta_max/num_theta)

r_min = 0.0
r_max = math.hypot(x_max,y_max)
rho_arr = np.arange(0,r_max+r_max/num_rho,r_max/num_rho) # Array with all possible values of rho

hough_space = np.zeros((num_rho,num_theta))  # Zero initialized hough space accumalator

for x in range(x_max):
    for y in range(y_max):
        # Check if the pixel value is a edge point or not
            # continue if pixel value is black since are edge image will be white lines on black
        if img[x,y,0]==0: 
            continue
        for thetai in range(num_theta):
            # looping through theta values and finding corresponding rho
            # and incrementing the accumalator
            theta = 1.0*thetai*theta_max/num_theta      # Increment*thetai
            rho = x*math.cos(theta)+y*math.sin(theta)   # Hough space function in terms of rho and theta
            # Find rhoi using rho
            rhoi = np.argmin(np.abs(rho_arr - rho))
            #Increment Accumalator
            hough_space[rhoi,thetai] += 1
        coord.append((x,y))
#print(len(coord))
#plt.imshow(hough_space, origin='lower',cmap='gray')
#plt.show()

""" Loop throught accumalator to find edge line """
lines = []
for y in range(hough_space.shape[0]):
    for x in range(hough_space.shape[1]):
        if hough_space[y][x] > threshold:
            # Find rho and theta
            rho = rho_arr[y]
            theta = theta_arr[x]
            
            # XY coord conversion
            m = -np.cos(np.deg2rad(theta))/np.sin(np.deg2rad(theta))        # m = -cot(theta)
            b = rho/np.sin(np.deg2rad(theta))                               # b = rho/sin(theta)
            # Find the point at the end of rho,theta vector
            x0 = rho*np.cos(np.deg2rad(theta))
            y0 = rho*np.sin(np.deg2rad(theta)) 
            lines.append((m,b))





#print(len(coord))
#plt.imshow(hough_space, origin='lower',cmap='gray')
#plt.show()

"""
for x in range(x_max):
    for y in range(y_max):
        # Check if the pixel value is a edge point or not
            # continue if pixel value is black since are edge image will be white lines on black
        if img[x,y,0]==0: 
            continue
        # put limits on a and b and then on x0 and y0
        a_max = max([x,x_max-x])/2
        b_max = max([y,y_max-y])/2
        
        x0_max = a_max
        y0_max = b_max
        y0_arr = np.arange(0,y0_max+y0_max/num_y0,y0_max/num_y0)

        # Create Hough Space for lame curve
        for ai in range(1,num_a):
            for bi in range(1,num_b):
                for x0i in range(1,num_x0):
                    a = 1.0*ai*a_max/num_a
                    b = 1.0*bi*b_max/num_b
                    x0 = 1.0*x0i*x0_max/num_x0
                    # find the corresponding y0
                    y0 = b*(1-(x0/a)**n)**(1/n)
                    # find y0 index for y0
                    y0i = np.argmin(np.abs(y0_arr - y0))
                    # update hough space
                    hough_space[x0i,y0i,ai,bi]+=1
print(hough_space.shape)
"""