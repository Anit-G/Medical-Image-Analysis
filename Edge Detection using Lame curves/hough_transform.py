import matplotlib.pyplot as plt
import numpy as np


"""
    Input:
        num_theta
        num_a
        num_b
        img
        threshold
        n

        Resourse used:
        https://www.codespeedy.com/implement-a-superellipse-in-python/
"""

"""print(para)
plt.figure()
plt.imshow(img)
t = np.linspace(0, 2 * np.pi, 100)
for a0,b0 in para:
    x = ((np.abs(np.cos(t))) ** (2 / n)) * a0 * np.sign(np.cos(t))
    y = ((np.abs(np.sin(t))) ** (2 / n)) * b0 * np.sign(np.sin(t))
    plt.plot(x,y)
plt.show()"""

"""__________________ Hough Space ___________________"""
def hough_transform(img,threshold=100,num_theta=1000,n=1.5):
    x_max = img.shape[0]
    y_max = img.shape[1]

    num_a = int(x_max/2)
    num_b = int(y_max/2)

    #Arrays of a, b, theta
    theta = np.linspace(-np.pi,np.pi,num_theta)
    a_arr = np.linspace(0,x_max/2,num_a)
    b_arr = np.linspace(0,y_max/2,num_b)

    hough_space = np.zeros((num_b,num_a))  # Zero initialized hough space accumalator

    fig,ax = plt.subplots(1,1,figsize=(12,12))
    ax.set_facecolor((0,0,0))

    for y in range(y_max):
        for x in range(x_max):
            # Check if the pixel value is a edge point or not
                # continue if pixel value is black since are edge image will be white lines on black
            if img[x,y]==0: 
                continue
            edge_point = [y-y_max/2,x-x_max/2]
            ys,xs=[],[]
            for thetai in range(num_theta):
                t = theta[thetai]
                cos,sin = np.cos(t), np.sin(t)
                b = edge_point[0]/(np.abs(sin)**0.5*np.sign(sin))
                a = edge_point[1]/(np.abs(cos)**0.5*np.sign(cos))

                #Ensure a and b are less then 1
                a_d = np.min(np.abs(a_arr-a))
                b_d = np.min(np.abs(b_arr-b))
                if not (a_d<1 and b_d<1):
                    continue
                ai = np.argmin(np.abs(a_arr-a))
                bi = np.argmin(np.abs(b_arr-b)) 

                # update accumalator
                hough_space[bi,ai]+=1
                ys.append(b)
                xs.append(a)
            # Plotting the hough space
            ax.plot(xs,ys,color='white',alpha=0.05)

    ax.set_title('Hough Space')
    fig.savefig('Hough Space.png')
    plt.clf()
    plt.close()
    
    para = []
    for bi in range(hough_space.shape[0]):
        for ai in range(hough_space.shape[1]):
            if hough_space[bi][ai] > threshold:
                # convert index to values
                a = a_arr[ai]       
                b = b_arr[bi]
                para.append((a,b))
    return para

