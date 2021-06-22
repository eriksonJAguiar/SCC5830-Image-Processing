import numpy as np
import imageio
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('../2019-scalingattack/scaleatt')

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
    
    def build_attack_tool(self, img_s, img_t):
        '''
            function to build attack using tool
            paramters:
                - image_s: source image
                - image_t: target image
            return:
                - A: attack image
        '''   
        from scaling.ScalingGenerator import ScalingGenerator
        from scaling.SuppScalingLibraries import SuppScalingLibraries
        from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
        from attack.QuadrScaleAttack import QuadraticScaleAttack
        from attack.ScaleAttackStrategy import ScaleAttackStrategy

        scaling_algorithm = SuppScalingAlgorithms.NEAREST
        scaling_library = SuppScalingLibraries.PIL

        scaler_approach = ScalingGenerator.create_scaling_approach(
            x_val_source_shape=img_s.shape,
            x_val_target_shape=img_t.shape,
            lib=scaling_library,
            alg=scaling_algorithm
        )

        scale_att =  QuadraticScaleAttack(eps=1, verbose=False)
        attack_image, _, _ = scale_att.attack(src_image=img_s,
                                             target_image=img_t,
                                             scaler_approach=scaler_approach)

        return attack_image
    
    def show_img(self, img, fname):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.imsave(fname, img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    
        
#img_s = imageio.imread('./chest.png', as_gray=True).astype(np.uint8)
#img_t = imageio.imread('./cat.jpg', as_gray=True).astype(np.uint8)
#img_attack = imageio.imread('./img_attack.png', as_gray=True).astype(np.uint8)
attack = ImageScaling()
attack.load_img('./img_attack.png')
#img_t = attack.nn_resize(img_attack, 128,128)
#img_resize = attack.b_resize(img, 128, 128)
#attack.show_img(img_resize)
#img_attack = attack.build_attack(img_s,img_t)
#attack.show_img(img_t, 'img_attack.png')
