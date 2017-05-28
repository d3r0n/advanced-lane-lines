
import numpy as np
import cv2

class Threshold:
    def blue_yellow(self,img_lab, min_max=(145, 255)):
        b_channel = img_lab[:,:,2]
        return self.__apply__(b_channel, min_max)

    def white(self, img_luv, min_max= (220, 255)):
        l_channel = img_luv[:,:,0]
        return self.__apply__(l_channel, min_max)

    def hue(self, img_hls, min_max = (0, 255)):
        h_channel = img_hls[:,:,0]
        return self.__apply__(h_channel, min_max)

    def saturation(self, img_hls, min_max = (0, 255)):
        s_channel = img_hls[:,:,2]
        return self.__apply__(s_channel, min_max)

    def lightness(self, img_hls, min_max = (0, 255)):
        l_channel = img_hls[:,:,1]
        return self.__apply__(l_channel, min_max)

    def gradient_magnitude_x(self, bit_channel, sobel_kernel=5, min_max = (0,255)):
        sobelx = cv2.Sobel(bit_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel = self.__magnitude_gradient__(sobelx)
        sobel = self.__scale_gradient__(sobel)
        return self.__apply__(sobel, min_max)

    def gradient_direction_xy(self, bit_channel, sobel_kernel=5, min_max = (0,np.pi/2)):
        sobelx = cv2.Sobel(bit_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(bit_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = self.__direction_gradient__(sobelx, sobely)
        return self.__apply__(sobel, min_max)

    def gradient_magnitude_xy(self, bit_channel, sobel_kernel=5, min_max = (0,255)):
        sobelx = cv2.Sobel(bit_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(bit_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = self.__magnitude_gradient__(sobelx, sobely)
        sobel = self.__scale_gradient__(sobel)
        return self.__apply__(sobel, min_max)

    def __magnitude_gradient__(self, sobel, extra_sobel = None):
        if extra_sobel is not None:
            return np.sqrt(sobel**2 + extra_sobel**2)
        else:
            return np.sqrt(sobel**2)

    def __direction_gradient__(self, sobelx, sobely):
        return np.arctan2(np.absolute(sobely), np.absolute(sobelx)) 

    def __scale_gradient__(self, sobel):
        scale_factor = np.max(sobel)/255
        return np.uint8(sobel/scale_factor)

    def __apply__(self, img_array, threshold):
        binary = np.zeros_like(img_array)
        binary[(threshold[0] <= img_array) & (img_array <= threshold[1])] = 1
        return binary
