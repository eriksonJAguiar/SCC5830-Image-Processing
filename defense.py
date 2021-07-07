import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import histogram2d
from scipy.signal import find_peaks
from scipy.ndimage import histogram
import sys


class Defense:

    def show_img(self, img, fname=None):
        """
            show image
        """
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if fname != None:
            plt.imsave(fname, img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()

    def resconstruct_image(self, img):
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
        
        return img_r

    def reconstruction_peak(self, img):
        """
            get local peaks and reconstruct image by using mean of neighborhood
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
    
    def plot_histogram(self, hist):
        """
            show histogram
        """
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.show()


if __name__ == '__main__':
    
    print('Load images ...')
    img_a = imageio.imread('./img_attack.png', as_gray=True).astype(np.uint8)

    print("Apply reconstruction ...")

    df = Defense()
    #hist, img_r = df.reconstruction_peak(img_a)
    img_r = df.resconstruct_image(img_a)
    df.show_img(img_r, fname="rec1-median.png")
    #print(peaks)
    #print(hist)
    #df.plot_histogram(hist)
    