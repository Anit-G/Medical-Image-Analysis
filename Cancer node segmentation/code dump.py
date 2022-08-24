def initial(r,a,b):
    # Initialize the Snake optimization
    s = np.linspace(0, 2*np.pi, 400)
    x = a + r*np.sin(s)
    y = b + r*np.cos(s)
    init = np.array([x, y]).T
    return init



from skimage.segmentation import active_contour
from skimage.filters import gaussian
fig, ax = plt.subplots(1,3,figsize=(25,10))

#Plot Images
ax[0].imshow(img1,cmap='gray')
ax[1].imshow(img2,cmap='gray')
ax[2].imshow(img3,cmap='gray')

#Plot initial circle
init = initial(150,350,350)
snake1 = active_contour(gaussian(img1_seg,5,preserve_range=False),init, alpha=0.015, beta=0.1, gamma=0.001)
ax[0].plot(init[:, 1], init[:, 0], '--r', lw=1)

init = initial(150,380,380)
snake2 = active_contour(gaussian(img2_seg,5,preserve_range=False), init, alpha=0.015, beta=0.1, gamma=0.001)
ax[1].plot(init[:, 1], init[:, 0], '--r', lw=1)

init = initial(70,270,180)
snake3 = active_contour(img3,init, alpha=0.01, beta=0.1, gamma=0.001)
ax[2].plot(init[:, 1], init[:, 0], '--r', lw=1)

#Plot snake
ax[0].plot(snake1[:, 1], snake1[:, 0], '-b', lw=3)
ax[1].plot(snake2[:, 1], snake2[:, 0], '-b', lw=3)
ax[2].plot(snake3[:, 1], snake3[:, 0], '-b', lw=3)

plt.show()




from skimage.segmentation import morphological_geodesic_active_contour,inverse_gaussian_gradient

fig, ax = plt.subplots(1,3,figsize=(25,10))

#Plot Images
ax[0].imshow(img1,cmap='gray')
ax[1].imshow(img2,cmap='gray')
ax[2].imshow(img3,cmap='gray')


# Initial level set
init_ls = np.zeros(img1.shape, dtype=np.int8)
init_ls[300:400, 300:410] = 1
# List with intermediate results for plotting the evolution
ls1 = morphological_geodesic_active_contour(inverse_gaussian_gradient(img1),iterations=230,
                                           init_level_set=init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69)
ax[0].contour(ls1)
#ax[0].set_title(cv.contourArea(ls1.astype('uint8')))

# Initial level set
init_ls = np.zeros(img2.shape, dtype=np.int8)
init_ls[350:400, 350:400] = 1
# List with intermediate results for plotting the evolution
ls2 = morphological_geodesic_active_contour(inverse_gaussian_gradient(img2),iterations=230,
                                           init_level_set=init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69)
ax[1].contour(ls2)
#ax[1].set_title(cv.contourArea(ls2.astype('uint8')))

# Initial level set
init_ls = np.zeros(img3.shape, dtype=np.int8)
init_ls[240:300, 150:200] = 1
# List with intermediate results for plotting the evolution
ls3 = morphological_geodesic_active_contour(img3.astype('uint8'),iterations=230,
                                           init_level_set=init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69)
ax[2].contour(ls3)
#[2].set_title(cv.contourArea(ls3.astype('uint8')))
