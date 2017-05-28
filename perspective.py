import numpy as np
import cv2

class Perspective:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.matrix, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
