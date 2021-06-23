import numpy as np
import imageio
import matplotlib.pyplot as plt
from numpy.random.mtrand import beta
from scipy.optimize import minimize
import math
import sys

from numpy.lib.utils import source
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
        r_img = np.zeros((m,n), dtype=np.uint8)
        im_m, im_n = img.shape
        img = np.pad(img, ((0,1),(0,1)), 'constant')

        for x in range(m):
            for y in range(n):

                sx = (x+1) * (im_m/m)-1
                sy = (y+1) * (im_n/n)-1

                i = math.floor(sx)
                j = math.floor(sy)

                u = sx - i
                v = sy - j

                r_img[x,y] = (1-u)*(1-v)*img[i,j]+u*(1-v)*img[i+1,j]+(1-u)*v*img[i,j+1]+u*v*img[i+1,j+1]

        return r_img

    def get_coeficient(self, beta, size_s, size_t):
        '''
            generate coeficients to create perturbations
            parameters:
                - beta: image factor
                - size_s height or width of source image
                - size_t: height or width of source image
               
            returns:
                - est_matrix: estimated matrix
        '''
        source = (beta * np.eye(size_s))
        output = self.b_resize(source, size_s, size_t)

        est_matrix = output/(np.sum(output, axis=1).reshape(size_t, 1))

        return est_matrix
    
    def get_perturbation(self, beta, mat_l, mat_r, img_s, img_t):
        '''
            get image perturbation
        '''
        #matrix height
        #mat_l = self.get_coeficient(m, m_l)
        #matrix width
        #mat_r = self.get_coeficient(n, n_l)
        m, n = img_s.shape
        m_l, n_l = img_t

        modif = np.zeros((m,n))

        attack = (np.tanh(modif) + 1)*0.5

        aux = attack.reshape(m,-1)
        aux = np.matmul(mat_l, aux)
        aux = np.transpose(aux, (1,0))
        aux = np.reshape(aux, (n, -1))
        aux = np.matmul(mat_r, aux)
        aux = np.reshape(aux, (-1, m_l))
        output = np.transpose(aux, (1,0))

        # perturbation delta 1
        d1 = attack - img_s
        #perturbation delta 2
        d2 = attack - img_t

        INmax = np.max(img_s)

        obj1 = np.sum(np.sqrt(d1))/(m*n)
        obj2 = (beta * np.sum(np.sqrt(d2)))/(m_l, n_l)
        obj = obj1 + obj2

        const = ({'type': 'ineq', 'func': self.constraint})

        attack_sol = minimize(self.objective, attack, method='', constraints=const, options={'disp': True})
        print(attack_sol)


    def objective(self, d1):
        '''
            objetive of optimization method
            parameters:
                - d: perturbation
            
        '''
        return np.power(np.linalg.norm(d1, ord='-inf'))
    
    def constraint(self, mat_l, img_s, imag_t, d1, INmax, epsilon, m,n):
        ''' 
            constraint of optimization method
        '''
        s_l = self.b_resize(img_s, m, n)
        return np.linalg.norm((mat_l*(s_l*d1) - imag_t), ord='-inf') >= epsilon*INmax
    
    def delta_noise(self, n,m):
        '''
            delta noise
        '''
        noise = np.random.logistic(1, 0.1, [n,m])

        return noise

    
    def build_attack(self, img_s, img_t):
        '''
            generate an attack image
            paramters:
                - image_s: source image
                - image_t: target image
            return:
                - A: attack image
        '''   
        m, n = img_s.shape
        m_l, n_l = img_t.shape
        beta = 1

        mat_l = self.get_coeficient(beta, m, m_l)
        mat_r = self.get_coeficient(beta, n, n_l)

        img_s = img_s/beta
        img_t = img_t/beta

        A = self.get_perturbation(beta, mat_l, mat_r, img_s, img_t)

        return (A*beta).astype(np.uint8)

    
    def show_img(self, img, fname):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        #plt.imsave(fname, img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    
        
img_s = imageio.imread('./chest.png', as_gray=True).astype(np.uint8)
img_t = imageio.imread('./cat.jpg', as_gray=True).astype(np.uint8)
#img_attack = imageio.imread('./img_attack.png', as_gray=True).astype(np.uint8)
attack = ImageScaling()
img_t = attack.b_resize(img_t, 128, 128)
attack_img = attack.build_attack(img_s, img_t)
#img_t = attack.nn_resize(img_attack, 128,128)
#img_resize = attack.b_resize(img, 128, 128)
#attack.show_img(img_resize)
#img_attack = attack.build_attack(img_s,img_t)
attack.show_img(attack_img, 'img_attack.png')
