import numpy as np
import imageio
import matplotlib.pyplot as plt
import cvxpy as opt
import math

#from numpy.lib.utils import source
#sys.path.append('../2019-scalingattack/scaleatt')

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

    def get_coeficient(self, m, n, ml, nl):
        '''
            generate coeficients to create perturbations
            parameters:
                - beta: image factor
                - size_s height or width of source image
                - size_t: height or width of source image
               
            returns:
                - est_matrix: estimated matrix
        '''
        S = (255 * np.identity(m))
        D = self.b_resize(S, ml, m).astype(np.uint)
        
        CL = D / 255
        i = np.arange(CL.shape[0])
        CL[i,:] = CL[i,:]/CL.sum()

        Sl = (255 * np.identity(n))
        Dl = self.b_resize(Sl, n, nl).astype(np.uint)

        CR = Dl / 255
        j = np.arange(CR.shape[1])
        CR[:,j] = CR[:,j] / CL.sum()

        return CL, CR

    def build_attack(self, img_s, img_t):
        '''
            build attack
            parameters: 
                - img_s: source image
                - img_t: target image
            return:
                - result_attack_img: attack image
        '''
        print('Building attack ...')
        m, n = img_s.shape
        ml, nl = img_t.shape
        print('Getting coeficients ...')
        CL, CR = self.get_coeficient(m,n, ml, nl)

        img_sl = self.b_resize(img_s, m, nl)

        print('Build one direction attack (CL) ...')
        attack_img1 = self.build_direct_attack(img_sl, img_t, CL)
        attack_img1 = np.clip(np.round(attack_img1), 0, 255)

        print('Build one direction attack (CR) ...')
        attack_img2 = self.build_direct_attack(img_s.T, attack_img1.T, CR.T)
        attack_img2 = np.clip(np.round(attack_img2), 0, 255)

        print('Building result image')
        result_attack_img = (attack_img2.T).astype(np.uint8)

        return result_attack_img


    def build_direct_attack(self, img_s, img_t, mat_coeficient):
        '''
            generate attack and get perturbation towards one direction (horizontal)
            paramters:
                - image_s: source image
                - image_t: target image
                - mat_coeficient: matrix for coeficient L or R
            return:
                - A: attack image
        '''   
        #A = np.copy(img_s).astype(np.float64)
        A = np.zeros(img_s.shape)
        only = np.arange(img_s.shape[0])
        #only = np.where(np.sum(mat_coeficient, axis=0))[0]

        #A = self.get_perturbation(beta, mat_l, mat_r, img_s, img_t)
        epsilon = 0.999

        optimal_values = np.zeros(img_s.shape[1])

        #run go horizontal
        print('Optimize perturbation ...')
        for h in range(img_s.shape[1]):
            source_h = img_s[only, h]
            target_h = img_t[:, h]

            optimal_prob, optimal_delta = None, None
            delta1 = opt.Variable(source_h.shape[0])
            mat_ident = np.identity(source_h.shape[0])
                
            cost = opt.quad_form(delta1, mat_ident)
            #cost = opt.sum_squares(delta1, mat_ident)/(img_s.shape[0]*img_s.shape[1])
            
            obj = opt.Minimize(cost)
            attack_img = (source_h + delta1)
            C_aux = mat_coeficient[:, only]
            aux = (C_aux @ attack_img) - target_h
            contrs = [attack_img <= 255, attack_img >= 0, opt.norm_inf(aux) <= epsilon*255]
            #constraint1 = attack_img <= 255
            #constraint2 = attack_img >= 0
            #C_aux = mat_coeficient[:, only]
            #constraint3 =  opt.norm_inf(C_aux @ attack_img - target_h) <= epsilon
            #constraint3 =  opt.abs(C_aux @ attack_img - target_h) <= epsilon
           

            optimal_prob = opt.Problem(obj, contrs)
            #optimal_prob.solve()
            

            probl_ind = self.probl_solve(optimal_prob, epsilon, h)
            optimal_delta = delta1

            if probl_ind is not True:
                raise Exception('Could not solve the problem')

            # if probl_ind is True:
            #     optimal_prob = probl
            #     optimal_delta = delta1
            #     if epsilon > highest_epsilon:
            #             highest_epsilon = ep
            #         break
            #     else:
            #         continue0
        
            if optimal_prob is None:
                raise Exception('Could not solve the problem')
            
            optimal_values[h] = optimal_prob.value

            assert optimal_delta is not None
            A[only, h] = source_h +  optimal_delta.value
    
        # if highest_epsilon > epsilon[0]:
        #     print('Another epsilon value was choose')


        return A

    def probl_solve(self, probl, epsilon, h):
        '''
            try to solve the proble by optimization method
            quadratic optimization
            parameters:
                - probl: problem to solve (library cvxy)
                - epsilon: epsilon testing
                - h: horizontal positions
            return:
                - bool: True if problem to solve , if not return False
        '''
        try:
            probl.solve()
        except Exception as e1:
            print('QSQP Solver failed')
            try:
                probl.solve(solver=opt.ECOS, verbose=True)
            except Exception as e2:
                print('Error to solve the problem: with epsilon {} and h {}'.format(epsilon, h))
                print('Errors: {} and {}'.format(e1, e2))
                return False
        
        if probl.status != opt.OPTIMAL and probl.status != opt.OPTIMAL_INACCURATE:
            print('Could not solve the problem')
            return False
        
        return True
    
    def adjust_target_image(self, img_s, img_t, new_size):
        '''
            this method convert image to another shape used by algorithm
            paramters:
                - img_s: source image
                - img_t target image
                - new_size: new shape of position 0 for target image
            return:
                - img_t_new: target image with new shapes
        '''
        _, n = img_s.shape
        img_t_new = attack.b_resize(img_t, new_size, n)

        return img_t_new
    
    def show_img(self, img, fname):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.imsave(fname, img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    
        

if __name__ == '__main__':
    print('Load images ...')
    img_s = imageio.imread('./chest.png', as_gray=True).astype(np.uint8)
    img_t = imageio.imread('./cat.jpg', as_gray=True).astype(np.uint8)

    attack = ImageScaling()
    print('Adjust target image...')
    img_t = attack.adjust_target_image(img_s, img_t, 128)
    print('Starting building attack ...')
    attack_img = attack.build_attack(img_s, img_t)
    print('Finish attack generation!')
    print('plot new image')
    attack.show_img(attack_img, 'img_attack_new.png')
