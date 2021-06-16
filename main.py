import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
import math

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

    def b_resize(self, img, n, m):
        '''
            function to resize image using Bilinear interpolation
            paramters:
                - img: source image
                - n: height of new image
                - m: width of new image
            return:
                - r_img: risized image (nxm)
        '''
        r_img = np.zeros((n,m), dtype=np.uint8)
        im_n, im_m = img.shape
        img = np.pad(img, ((0,1),(0,1)), 'constant')

        for x in range(n):
            for y in range(m):

                sx = (x+1) * (im_n/n)-1
                sy = (y+1) * (im_m/m)-1

                i = math.floor(sx)
                j = math.floor(sy)

                u = sx - i
                v = sy - j

                r_img[x,y] = (1-u)*(1-v)*img[i,j]+u*(1-v)*img[i+1,j]+(1-u)*v*img[i,j+1]+u*v*img[i+1,j+1]

        return r_img
    
    def show_img(self, img):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

        plt.show()
        


img = imageio.imread('./chest.png', as_gray=True)
attack = ImageScaling()
#img_resize = attack.nn_resize(img, 128,128)
img_resize = attack.b_resize(img, 128, 128)
attack.show_img(img_resize)
