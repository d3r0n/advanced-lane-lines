import numpy as np
import cv2

class Perspective:
    def __init__(self, warp_x_size, warp_y_size, top_left, top_right, bottom_right, bottom_left):
        self.src = np.float32([top_left, top_right, bottom_right, bottom_left])
        self.dst = self.__dst__(warp_x_size, warp_y_size)
        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)

    def __dst__(self, x, y):
        offset = 0
        return np.float32([[offset, offset],
                         [y-offset, offset],
                         [y-offset, x-offset],
                         [offset, x-offset]])

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.matrix, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
