import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

class ImageScaling:
    '''
        Image scalling attacks methods for images
    '''
    def nn_resize(self, img, n, m):
        '''
            function to resize image using Nearest-neighbor interpolation
            paramters:
                - img: source image
                - n: height of new image
                - m: width of new image
            return:
                - r_img: risized image (nxm) 
        '''
        N, M = img.shape
        r_img = np.zeros((n,m), dtype=img.dtype)
        for x in range(n-1):
            for y in range(m-1):
                scx = round(x*(N/n))
                scy = round(y*(M/m))
                r_img[x,y] = img[scx, scy]

        return r_img
    
    def show_img(self, img):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()
        


img = imageio.imread('chest.png', as_gray=True)
attack = ImageScaling()
img_reized = attack.nn_resize(img, 128,128)
attack.show_img(img_reized)
