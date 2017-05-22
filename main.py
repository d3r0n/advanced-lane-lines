# %% IMPORTS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from collections import deque
# %matplotlib inline

import camera
from threshold import Threshold
# from lane_search import HistogramSearch, LaneDetection
from perspective import Perspective

# %% CALIBRATION
rms, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = camera.calibration()

# %% PIPELINE
class HistogramSearch:
    def __init__(self, window_width = 200, window_height = 80, min_pixels_to_recenter=50, visualisation = None):
        self.margin = np.int(window_width / 2)
        self.window_height = window_height
        self.min_pixels_to_recenter = min_pixels_to_recenter
        self.visualisation = visualisation

    def pixel_histogram(self, img):
        return np.sum(img[img.shape[0]//2:,:], axis=0)

    def __find_start_points__(self, img):
        histogram = self.pixel_histogram(img)
        midpoint = np.int(histogram.shape[0]/2)
        left_x_start = np.argmax(histogram[:midpoint])
        right_x_start = np.argmax(histogram[midpoint:]) + midpoint
        return (left_x_start, right_x_start)

    def __find_pixel_indices_in_window__(self, x_current , y_current, nonzerox, nonzeroy):
        window_y_low = y_current - self.window_height
        window_y_high = y_current
        window_x_low = x_current - self.margin
        window_x_high = x_current + self.margin

        if self.visualisation is not None:
            cv2.rectangle(self.visualisation,(window_x_low,window_y_low),(window_x_high,window_y_high),(0,255,0), 4)

        tmp = (nonzeroy >= window_y_low) \
            & (nonzeroy < window_y_high) \
            & (nonzerox >= window_x_low) \
            & (nonzerox < window_x_high)

        return tmp.nonzero()[0]

    def __lane_pixels__(self, x_start, y_start, nonzerox, nonzeroy):
        x_current = x_start
        y_current = y_start

        lane_pixels = []
        n_windows = np.int(y_start / self.window_height)
        for window in range(n_windows):
                window_pixels = self.__find_pixel_indices_in_window__(x_current, y_current, nonzerox, nonzeroy)
                lane_pixels.append(window_pixels)
                if len(window_pixels) > self.min_pixels_to_recenter:
                    x_current = np.int(np.mean(nonzerox[window_pixels]))
                y_current -= self.window_height

        return np.concatenate(lane_pixels)

    def __lane_pixels_near_previous_fit__(self, fit, nonzerox, nonzeroy):
        return ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - self.margin)) \
            &   (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + self.margin)))

    def find_lane_polynomials(self, img, prev_left_fit = None, prev_right_fit = None):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Find lane pixel indices
        if prev_left_fit is not None and prev_right_fit is not None:
            left_lane_pixels = self.__lane_pixels_near_previous_fit__(prev_left_fit, nonzerox, nonzeroy)
            right_lane_pixels = self.__lane_pixels_near_previous_fit__(prev_right_fit, nonzerox, nonzeroy)
        else:
            left_x_start, right_x_start = self.__find_start_points__(img)
            left_lane_pixels = self.__lane_pixels__(left_x_start, img.shape[0], nonzerox, nonzeroy)
            right_lane_pixels = self.__lane_pixels__(right_x_start, img.shape[0], nonzerox, nonzeroy)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_pixels]
        lefty = nonzeroy[left_lane_pixels]
        rightx = nonzerox[right_lane_pixels]
        righty = nonzeroy[right_lane_pixels]

        if self.visualisation is not None:
            self.visualisation[lefty,leftx] = [255, 0, 0]
            self.visualisation[righty,rightx] = [0, 0, 255]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_detection = LaneDetection(left_fit, leftx, lefty)
        right_detection = LaneDetection(right_fit, rightx, righty)

        return (left_detection, right_detection)

    def visualisation_elements(self, img):
        # making the original road pixels 3 color channels
        gr_img = img * 255
        self.visualisation = np.array(cv2.merge([gr_img,gr_img,gr_img]),np.uint8)
        left_detection, right_detection  = self.find_lane_polynomials(img)
        left_fit, right_fit = left_detection.fit, right_detection.fit

        # Generate x and y values for plotting
        arguments_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_values_x = left_fit[0]*arguments_y**2 + left_fit[1]*arguments_y + left_fit[2]
        right_values_x = right_fit[0]*arguments_y**2 + right_fit[1]*arguments_y + right_fit[2]

        return (self.visualisation, left_values_x, right_values_x, arguments_y)

class LaneDetection:
    def __init__(self, fit, detected_x, detected_y):
        self.fit = fit
        self.detected_x = detected_x
        self.detected_y = detected_y

class Lane:
    def __init__(self, n_detections):
        self.n = n_detections
        # x values of the last n fits of the line
        self.detections = deque()
        #average x values of the fitted line over the last n iterations
        self.best_x = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #difference in fit coefficients between last and new fits
        self.last_coef_diff = np.array([0,0,0], dtype='float')

    def update(self, lane_detection):
        self.detections.append(lane_detection)
        if len(self.detections) > self.n: self.detections.popleft()
        self.best_fit = self.__smooth_fits__(self.detections)

        return self

    def __smooth_fits__(self, queue):
        α = len(queue) / 100 + 0.05
        head, *tail = queue
        res_a, res_b ,res_c = head.fit
        for detection in tail:
            a, b, c = detection.fit
            α -= 0.01
            β = 1.0 - α
            res_a = (res_a * α) + (a * β)
            res_b = (res_b * α) + (b * β)
            res_c = (res_c * α) + (c * β)
        # queue.append(((alpha+res_alpha)/2, (beta + res_beta)/2))
        return np.array([res_a, res_b, res_c])

class LaneStatistics:
    def __init__(self, captured_image, lane):
        self.meters_per_pixel_verticaly = 30/720
        self.meters_per_pixel_horizontaly = 3.7/700

        real_world_fit = self.__real_world_fit__(lane.best_fit, captured_image.shape[0])
        self.radius_of_curvature_in_m = self.__evaluate_radius_of_curvature__(captured_image.shape[0], real_world_fit)
        self.distance_to_vehicle_center_in_m =self.__evaluate_distance_to_vehicle_center(captured_image, real_world_fit)

    def __real_world_fit__(self, fit, captured_image_height):
        arguments_y = np.linspace(0, captured_image_height-1, captured_image_height)
        values_x = fit[0]*arguments_y**2 + fit[1]*arguments_y + fit[2]

        return np.polyfit(arguments_y * self.meters_per_pixel_verticaly,
                          values_x * self.meters_per_pixel_horizontaly, 2)

    def __evaluate_distance_to_vehicle_center(self, captured_image, fit):
        center_y = captured_image.shape[0]*self.meters_per_pixel_verticaly
        center_x = captured_image.shape[1]/2 * self.meters_per_pixel_horizontaly
        evaluation_point_y = captured_image.shape[0] * self.meters_per_pixel_verticaly
        evaluation_point_x = (fit[0]*evaluation_point_y**2 + fit[1]*evaluation_point_y + fit[2])
        return np.linalg.norm(np.array([center_y, center_x]) - np.array([evaluation_point_y, evaluation_point_x]))

    def __evaluate_radius_of_curvature__(self, captured_image_height, fit):
        row_close_to_car_in_m = captured_image_height * self.meters_per_pixel_verticaly
        return ((1 + (2*fit[0]*row_close_to_car_in_m + fit[1])**2)**1.5) / np.absolute(2*fit[0])

class LaneLines:
    def __init__(self, camera_matrix, distortion_coefficients, buffer_size = 50):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.perspective = Perspective(
                            warp_x_size= 700,
                            warp_y_size= 1200,
                            top_left= (530,475),
                            top_right=(800,475),
                            bottom_right=(1280, 700),
                            bottom_left=(100, 700))
        self.search_engine = HistogramSearch(
                            window_width = 200,
                            window_height = 80,
                            min_pixels_to_recenter=50,
                            visualisation = None)
        self.left = Lane(buffer_size)
        self.right = Lane(buffer_size)

    def draw(self, img):
        preprocessed_img = self.preprocess(img)
        left, right = self.find_lanes(preprocessed_img)
        area = self.lane_area(preprocessed_img, left, right)
        area = cv2.warpPerspective(area, self.perspective.matrix, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        return cv2.addWeighted(img, 1, area, 0.3, 0)

    def find_lanes(self, preprocessed_img):
        l_detection, r_detection = self.search_engine.find_lane_polynomials(
                    preprocessed_img,
                    self.left.best_fit,
                    self.right.best_fit)
        self.left.update(l_detection)
        self.right.update(r_detection)
        return (self.left, self.right)

    def lane_area(self, warped, left_lane, right_lane):
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        arguments_y = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_values_x = left_lane.best_fit[0]*arguments_y**2 + left_lane.best_fit[1]*arguments_y + left_lane.best_fit[2]
        right_values_x = right_lane.best_fit[0]*arguments_y**2 + right_lane.best_fit[1]*arguments_y + right_lane.best_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_values_x, arguments_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_values_x, arguments_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        return color_warp

    def preprocess(self, img):
        img = self.undistort(img)
        img = self.blur(img) # ??
        img = self.threshold(img)
        img = self.warp(img)
        return img

    def warp(self, img):
        return self.perspective.warp(img)

    def blur(self, img,  kernel_size = 5):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def undistort(self, img):
        # h,  w = img.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        return  cv2.undistort(img, self.camera_matrix, self.distortion_coefficients, None, self.camera_matrix)

    def threshold(self, img):
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        threshold = Threshold()
        c_binary = threshold.color(img_hls, min_max = (170, 255))
        g_binary = threshold.gradient(img_hls, sobel_kernel=5, min_max = (20,100))
        img_binary = np.clip(c_binary + g_binary, 0 , 1)
        return img_binary

# %% RUNVIDEO
from moviepy.editor import VideoFileClip
from IPython.display import HTML

lane_lines = LaneLines(camera_matrix, distortion_coefficients)
output = 'challenge_output.mp4'
clip1 = VideoFileClip('challenge_video.mp4')
white_clip = clip1.fl_image(lane_lines.draw)
%time white_clip.write_videofile(output, audio=False)

# %% DEBUG
test_dir = 'test/'
for image_src in os.listdir(test_dir):
    img = cv2.imread(test_dir + image_src) #BRG colorspace
    lane_lines = LaneLines(camera_matrix, distortion_coefficients, 1)
    preprocessed_img = lane_lines.preprocess(img)
    l, r = lane_lines.find_lanes(preprocessed_img)
    lrc = LaneStatistics(img, l)
    rrc = LaneStatistics(img, r)
    print(lrc.radius_of_curvature_in_m,"m", rrc.radius_of_curvature_in_m, "m\t | ", lrc.distance_to_vehicle_center_in_m, "m", rrc.distance_to_vehicle_center_in_m, "m")

# %% PLOT HISTOGRAM OF PREPROCESSED TEST IMAGE
lane_lines = LaneLines(camera_matrix, distortion_coefficients)
img = cv2.imread('test_images/test6.jpg')
img = lane_lines.preprocess(img)
h_search = HistogramSearch()
histogram = h_search.pixel_histogram(img)
plt.plot(histogram)

# %% RUN PIPELINE ON TEST IMAGES
test_dir = 'test/'
output_dir = 'output_images/'
for image_src in os.listdir(test_dir):
    img = cv2.imread(test_dir + image_src) #BRG colorspace
    lane_lines = LaneLines(camera_matrix, distortion_coefficients)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    f.tight_layout()

    result = lane_lines.draw(img)
    b,g,r = cv2.split(result)           # get b,g,r
    rgb_result = cv2.merge([r,g,b])     # switch it to rgb
    ax1.imshow(rgb_result)
    #plot warp srource
    # src = lane_lines.perspective.src
    # ax1.plot(src[0,0], src[0,1], '.')
    # ax1.plot(src[1,0], src[1,1], '.')
    # ax1.plot(src[2,0], src[2,1], '.')
    # ax1.plot(src[3,0], src[3,1], '.')
    ax1.set_title('Cast of found lane area on to input image')
    ax1.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    preprocessed_img = lane_lines.preprocess(img)
    (visualisation, left_values_x, right_values_x, arguments_y) = lane_lines.search_engine.visualisation_elements(preprocessed_img)
    ax2.imshow(visualisation)
    ax2.plot(left_values_x, arguments_y, color='yellow')
    ax2.plot(right_values_x, arguments_y, color='yellow')
    ax2.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax2.set_title('Histogram search in warp space')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
    plt.savefig(output_dir + image_src)
