
import numpy as np
import cv2

class Threshold:
    def color(self, img_hls, min_max = (0, 255)):
        s_channel = img_hls[:,:,2]
        return self.__apply__(s_channel, min_max)

    def gradient(self, img_hls, sobel_kernel=5, min_max = (0,255)):
        l_channel = img_hls[:,:,1]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        # sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel = self.magnitude_gradient(sobelx)
        sobel = self.scale_gradient(sobel)
        return self.__apply__(sobel, min_max)

    def magnitude_gradient(self, sobel, extra_sobel = None):
        if extra_sobel is not None:
            return np.sqrt(sobel**2 + extra_sobel**2)
        else:
            return np.sqrt(sobel**2)

    def direction_gradient(self, sobelx, sobely):
        return np.arctan2(np.absolute(sobely), np.absolute(sobelx)) #example threshold (0.7, 1.3)

    def scale_gradient(self, sobel):
        scale_factor = np.max(sobel)/255
        return np.uint8(sobel/scale_factor)

    def canny(self, img_hls, min_max = (0, 255)):
        l_channel = np.uint8(img_hls[:,:,1])
        return cv2.Canny(l_channel, min_max[0], min_max[1])

    def __apply__(self, img_array, threshold):
        binary = np.zeros_like(img_array)
        binary[(threshold[0] <= img_array) & (img_array <= threshold[1])] = 1
        return binary
