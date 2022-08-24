import numpy as np
def swap_update(swap,x,y,c_type,qn):
    #1: cut along beta
    #2: cut along alpha
    #3: cross cut 1 = pa,qb,n = \
    #4: cross cut 2 = pb,qa,n = /
    # qn = [[1,2],[2,3],[3,2],[2,1]]

    update = [[1,1],[0,0],[0,1],[1,0]]
    print(c_type)
    p,q = update[c_type]
    swap[x+qn[0],y+qn[1],1] = q
    swap[x+2,y+2,1] =  p
    
    swap[x+qn[0],y+qn[1],0] = 1
    swap[x+2,y+2,0] =  1

    return swap


def maxflow(kernel,x,y,swap):
    assert kernel.shape[0]==kernel.shape[1]
    assert kernel.shape[0]==3
    # Alpha label = 1
    # Beta label = 0
    swap_k = swap[x:x+3,y:y+3,0]
    swap_ab = swap[x:x+3,y:y+3,1]
    
    # center piece of the kernel
    ip = kernel[2,2]
    # q is defined as N,E,S,W. Clockwise rotation
    print(2)
    iq = np.array([kernel[1,2],kernel[2,3],kernel[3,2],kernel[2,1]])
    print(1)
    qn = np.asarray([[1,2],[2,3],[3,2],[2,1]])

    Da = (iq-1)**2
    Db = iq**2
    cuts = []
    cuts_i = []

    for i,n,q in zip([0,1,2,3],qn,iq):
        # check t-link connections of p and q
        V = np.exp(-0.5*(ip-q)**2)
        if int(swap_k[2,2]+swap_k[n[0],n[1]])==4:
            #Vpq = 0 as fp and fq are the same

            #1: cut along beta
            #2: cut along alpha
            #3: cross cut 1 = pa,qb,n = \
            #4: cross cut 2 = pb,qa,n = /
            C = max((Db[i] + (ip)**2),(Da[i]+(ip-1)**2),((ip-1)**2+Db[i]+V),((ip)**2+Da[i]+V))
            ic = np.argmax((Db[i] + (ip)**2),(Da[i]+(ip-1)**2),((ip-1)**2+Db[i]+V),((ip)**2+Da[i]+V))
            
        elif int(swap_k[2,2]+swap_k[n[0],n[1]])==3:
            #check if p has 2 connections or q does
            if int(swap_k[2,2])==2:
                # p has 2 connections
                if swap_ab[n[0],n[1]] == 1:
                    C = max((ip**2),((ip-1)**2+V))
                    ic = np.argmax((ip**2),((ip-1)**2+V))

                else:
                    C = max(((ip-1)**2),((ip)**2+V))
                    ic = np.argmax(((ip-1)**2),((ip)**2+V))
            else:
                # q has 2 connections
                if swap_ab[2,2] == 1:
                    C = max((Db[i]),(Da[i]+V))
                    ic = np.argmax((Db[i]),(Da[i]+V))

                else:
                    C = max((Da[i]),(Db[i]+V))
                    ic = np.argmax((Da[i]),(Db[i]+V))

        elif int(swap_k[2,2]+swap_k[n[0],n[1]])==2:
            continue
        else:
            None
        
        cuts.append(C)
        cuts_i.append(ic)
    
    #cut = max(cuts)
    cut_ic = cuts_i[np.argmax(cuts)]
    cut_qn = qn[np.argmax(cuts)]

    swap = swap_update(swap,x,y,cut_ic,cut_qn)
    return swap


def convolve2D(image,swap=None, padding=1, strides=3):
    assert len(image.shape)==2
    # Swap record
    if swap==None:
        swap = 2*np.ones((image.shape[0],image.shape[1],2),dtype=int)

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2),dtype=int)
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        
        swap = 2*np.ones((image.shape[0]+padding*2,image.shape[1]+padding*2,2),dtype=int)

        
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1]:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0]:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        #maxflow enery function
                        kernel = imagePadded[x: x + 3, y: y + 3]
                        swap = maxflow(kernel,x,y,swap)
                except:
                    break

    return swap