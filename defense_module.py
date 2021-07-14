import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import histogram2d
from scipy.signal import find_peaks
from scipy.ndimage import histogram
import sys
import scipy

class Defense:
    """Defensive methods propose"""

    def cos_dist(self, hist1, hist2):
        """
            calculate similarity of two histograms
            parameters:
                - hist1: hotogram of source image
                - hist2: histogram of target image
            return
                - dist: distance between two histogram
        """
        h1_norm = np.sqrt(np.sum(hist1**2))
        h2_norm = np.sqrt(np.sum(hist2**2))
        sim  = np.sum(hist1*hist2)/(h1_norm*h2_norm)

        return sim
    
    def psrn(self, original, perturb):
        """
            measure diffence between two modified images
            parameters:
                - original: original image
                - perturb: crafted image
            return:
                - psrn: Peak Signal to Noise Ratio in dB
        """
        n,m = original.shape
        psrn = 10*np.log10(255/((1/(n*m)*np.sum((original - perturb)**2))))

        return psrn 

    def show_img(self, img, fname=None):
        """
            plot image
            paramters:
                - img: source image
                - fname: file name to save
        """
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if fname != None:
            plt.imsave(fname, img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    
    def restoration_image_median(self, img):
        """
            reconstruct image by median filter
            paramters:
                - img: attack image
            retun:
                - img_r: reconstruct image
        """
        n, m = 4,4
        N,M = img.shape

        a = int((n-1)/2)
        b = int((m-1)/2)

        img_r = np.zeros((N,M), dtype=np.uint8)
        img_pad = np.pad(img, (a,b), mode='symmetric')
    
        for x in range(a,N+a):
            for y in range(b, M+b):
                f_sub = img_pad[x-a:x+a+1, y-b:y+b+1]
                img_r[x-a,y-b] = np.median(f_sub)
        
        hist_new = self.build_histogram(img_r, 256)
        
        return hist_new, img_r

    def restoration_image_min(self, img):
        """
            reconstruct image by minimum filter
            paramters:
                - img: attack image
            retun:
                - img_r: reconstruct image
        """
        n, m = 4,4
        N,M = img.shape

        a = int((n-1)/2)
        b = int((m-1)/2)

        img_r = np.zeros((N,M), dtype=np.uint8)
        img_pad = np.pad(img, (a,b), mode='symmetric')
    
        for x in range(a,N+a):
            for y in range(b, M+b):
                f_sub = img_pad[x-a:x+a+1, y-b:y+b+1]
                img_r[x-a,y-b] = np.min(f_sub)
        
        hist_new = self.build_histogram(img_r, 256)
        
        return hist_new, img_r
    
    def restoration_peak(self, img):
        """
            get local peaks and reconstruct image by using mean of neighborhood
            paramters:
                - img: attack image
            retun:
                - hist_new: hitogram of new image
                - img_r: reconstruct image

        """
        N, M = img.shape
        hist, bins = np.histogram(img, bins=np.arange(0,255), density=True)
        peaks = find_peaks(hist)[0]

        a, b = 1, 1
        img_r = img.copy()
        img_pad = np.pad(img, (a,b), mode='symmetric')
    
        for pos in peaks:
            indices = np.where(img_r == bins[pos].astype(np.int))
            for i,j in zip(indices[0], indices[1]):
                img_r[i,j] = np.median(img_r[i:i+4, j:j+4])
        
        hist_new = np.histogram(img_r, bins=256)[0]
        
        return hist_new, img_r
    
    def unsharp_mask(self, img):
        """ 
            apply filter unsharp mask
        """
        #1 - apply gaussian blur
        N,M = img.shape
        g_size = 5

        g_filter = np.zeros([g_size,g_size], dtype=np.float)

        n, m = g_size//2, g_size//2
        sigma = 1
        for x in range(-m, m+1):
            for y in range(-n, n+1):
                g_filter[x+m, y+n] = (1/2*np.pi*(sigma**2))*np.exp(-(x**2 + y**2)/(2* sigma**2))
        
        a = int((g_size-1)/2)
        b = int((g_size-1)/2)

        img_blur = np.zeros((N,M), dtype=np.uint8)
        img_pad = np.pad(img, (a,b), mode='symmetric')
    
        for x in range(a,N+a):
            for y in range(b, M+b):
                f_sub = img_pad[x-a:x+a+1, y-b:y+b+1]
                img_blur[x-a,y-b] = np.sum(f_sub*g_filter)
        
        
        mask = img - img_blur
        img_r = img + mask
        img_r = np.clip(img_r, 0, 255).astype(np.uint8)
        
        hist_new = self.build_histogram(img_r, 256)
        
        return hist_new, img_r
    
    def build_histogram(self, img, levels):
        """
            generate histogram
            parameters:
                - img: reference image
                - levels: number of bins
            return 
                - hist: histogram of image
        """
        N, M = img.shape
        hist = np.zeros(levels).astype(int)
    
        for l in range(levels):
            px = np.sum(img == l)
            hist[l] = px
            
        return hist

    def plot_histogram(self, hist):
        """
            show histogram
            parameter:
                - hist: image histogram
        """
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.show()
