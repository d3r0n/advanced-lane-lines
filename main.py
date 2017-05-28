# %% IMPORTS
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from collections import deque
%matplotlib inline

import camera
from threshold import Threshold
from lane_search import HistogramSearch, LaneDetection
from perspective import Perspective

# %% CALIBRATION
rms, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = camera.calibration()

# %% PIPELINE
class Lane:
    def __init__(self, n_detections):
        self.n = n_detections
        self.detections = deque()
        self.best_x = None
        self.best_fit = None
        self.__prev_avg_fits__ = deque()

    def update(self, lane_detection):
        if lane_detection is not None:
            self.best_fit = self.__smooth_fits__(lane_detection)
            self.detections.append(lane_detection)
            if len(self.detections) > self.n:
                self.detections.popleft()

    def __smooth_fits__(self, new_detection):
        if len(self.detections) == 0: return new_detection.fit
        α = 0.5
        res_a, res_b, res_c = new_detection.fit
        for det in reversed(self.detections):
            β = 1.0 - α
            if β < 0.05: break
            a, b, c = det.fit
            res_a = (res_a * α) + (a * β)
            res_b = (res_b * α) + (b * β)
            res_c = (res_c * α) + (c * β)
            α += 0.03
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
    def __init__(self, camera_matrix, distortion_coefficients, buffer_size = 50, img_size = (1280,720)):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        src = np.float32(
            [[(img_size[0] / 2) - 75, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 75), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])

        self.perspective = Perspective(src,dst)

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
        result = img
        if left.best_fit is not None and right.best_fit is not None:
            area = self.lane_area(preprocessed_img, left, right)
            area = cv2.warpPerspective(area, self.perspective.matrix, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
            result = cv2.addWeighted(img, 1, area, 0.3, 0)

            ls = LaneStatistics(preprocessed_img, left)
            rs = LaneStatistics(preprocessed_img, right)

            radius = "radius of curvature {0:.0f} m"  \
                .format((ls.radius_of_curvature_in_m + rs.radius_of_curvature_in_m)/2)

            offset = "offset {0:.2f} m" \
                .format(np.abs(ls.distance_to_vehicle_center_in_m - rs.distance_to_vehicle_center_in_m))

            cv2.putText(result, radius, (50, 65), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255),2);
            cv2.putText(result, offset, (50, 135), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255),2);

        return result

    def find_lanes(self, preprocessed_img):
        l_detection, r_detection = self.search_engine.find_lane_polynomials(
                    preprocessed_img,
                    self.left.best_fit,
                    self.right.best_fit)

        if l_detection is None or r_detection is None or not l_detection.trustful() or not r_detection.trustful():
            l_detection, r_detection = self.search_engine.find_lane_polynomials(preprocessed_img)

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
        img = self.blur(img)
        img = self.threshold(img)
        img = self.warp(img)
        return img

    def warp(self, img):
        return self.perspective.warp(img)

    def blur(self, img,  kernel_size = 5):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def undistort(self, img):
        return  cv2.undistort(img, self.camera_matrix, self.distortion_coefficients, None, self.camera_matrix)

    def threshold(self, img):
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float)
        img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV).astype(np.float)

        threshold = Threshold()

        w_binary = threshold.white(img_luv, min_max= (180, 255))
        l_x = threshold.gradient_magnitude_x(img_luv[:,:,0], sobel_kernel=5, min_max = (15,255))
        l_d = threshold.gradient_direction_xy(img_luv[:,:,0], sobel_kernel=3, min_max = (0.7, 1.3))
        white = np.zeros_like(w_binary)
        white[(img[:,:,0] >= 200) & (img[:,:,1] >= 200) & (img[:,:,2] >= 200)] = 1
        r_x = threshold.gradient_magnitude_x(white, sobel_kernel=5, min_max = (10,255))
        l_binary = threshold.lightness(img_hls, min_max = (210, 255))
        g_m = threshold.gradient_magnitude_x(l_binary, sobel_kernel=31, min_max = (100, 255))

        white_lane = np.zeros_like(w_binary)
        white_lane [(r_x == 1) | ((l_d == 1) & (l_x == 1) & (w_binary == 1)) | (g_m == 1)] = 1

        y_binary = threshold.blue_yellow(img_lab, min_max= (145, 255))
        y_m = threshold.gradient_magnitude_xy(img_lab[:,:,2], sobel_kernel=3, min_max = (50, 255))
        s_binary = threshold.saturation(img_hls, min_max = (10,255))
        s_x = threshold.gradient_magnitude_x(img_hls[:,:,2], sobel_kernel=31, min_max = (100,255))
        s_d = threshold.gradient_direction_xy(img_hls[:,:,2], sobel_kernel=31, min_max = (0.6, 1.2))

        yellow_line = np.zeros_like(y_binary)
        yellow_line[(y_binary == 1) | (y_m == 1) | ((s_d == 1) & (s_x == 1) & (s_binary == 1))] = 1

        result = np.zeros_like(w_binary)
        result[(yellow_line == 1) | (white_lane == 1)] = 1
        return result

# %% UNDISTORT IMAGE
lane_lines = LaneLines(camera_matrix, distortion_coefficients)
img = cv2.imread('test_images/test5.jpg')
img = lane_lines.undistort(img)
cv2.imwrite('undistorted_test.jpg', img)

# %% CALCULATE CURVATURE AND OFFSET
test_dir = 'test_images/'
output_dir = 'output_curvature/'
for image_src in os.listdir(test_dir):
    img = cv2.imread(test_dir + image_src) #BRG colorspace
    lane_lines = LaneLines(camera_matrix, distortion_coefficients, 1)

    preproc = lane_lines.preprocess(img) * 255
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    preproc_3c = np.array(cv2.merge([preproc, preproc, preproc]),np.uint8)
    ax1.imshow(preproc_3c)

    src = lane_lines.perspective.src
    ax2.plot(src[0,0], src[0,1], '.')
    ax2.plot(src[1,0], src[1,1], '.')
    ax2.plot(src[2,0], src[2,1], '.')
    ax2.plot(src[3,0], src[3,1], '.')
    b,g,r = cv2.split(img)
    rgb_result = cv2.merge([r,g,b])
    ax2.imshow(rgb_result)

    l, r = lane_lines.find_lanes(preproc)
    rrc = LaneStatistics(img, r)
    lrc = LaneStatistics(img, l)

    info_left = " left radius {0:.0f} m \n right radius {0:.0f} m" \
                .format(lrc.radius_of_curvature_in_m, rrc.radius_of_curvature_in_m)

    plt.text(0.5, 0.2, info_left,
     horizontalalignment='center',
     verticalalignment='center',
     color='white',
     transform = ax1.transAxes)

    info_right = "offset {0:.2f} m" \
                .format(np.abs(lrc.distance_to_vehicle_center_in_m - rrc.distance_to_vehicle_center_in_m))

    plt.text(0.5, 0.2, info_right,
     horizontalalignment='center',
     verticalalignment='center',
     color='white',
     transform = ax2.transAxes)
    plt.savefig(output_dir + image_src)

# %% PLOT HISTOGRAM OF PREPROCESSED TEST IMAGE
lane_lines = LaneLines(camera_matrix, distortion_coefficients)
img = cv2.imread('test_images/video4.png')
img = lane_lines.preprocess(img)
h_search = HistogramSearch()
histogram = h_search.pixel_histogram(img)
plt.plot(histogram)
plt.savefig("histogram")

# %% RUN PIPELINE ON TEST IMAGES
test_dir = 'test_images/'
output_dir = 'output_images/'
for image_src in os.listdir(test_dir):
    img = cv2.imread(test_dir + image_src) #BRG colorspace
    lane_lines = LaneLines(camera_matrix, distortion_coefficients)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    f.tight_layout()

    result = lane_lines.draw(img)
    b,g,r = cv2.split(result)
    rgb_result = cv2.merge([r,g,b])
    ax1.imshow(rgb_result)
    ax1.set_title('Cast of found lane area on the input image')
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

# %% RUNVIDEO
from moviepy.editor import VideoFileClip
from IPython.display import HTML

lane_lines = LaneLines(camera_matrix, distortion_coefficients)
output = 'project_output.mp4'
clip1 = VideoFileClip('project_video.mp4')
def step(img):

    r,g,b = cv2.split(img)
    bgr_img = cv2.merge([b,g,r])
    lines = lane_lines.draw(bgr_img)
    b,g,r = cv2.split(lines)
    rgb_lines = cv2.merge([r,g,b])

    preproc = lane_lines.preprocess(bgr_img) * 255
    preproc_3c = np.array(cv2.merge([preproc, preproc, preproc]),np.uint8)
    p_small = cv2.resize(preproc_3c, (0,0), fx=0.5, fy=0.5)

    bird = lane_lines.undistort(img)
    bird = lane_lines.warp(bird)
    b_small = cv2.resize(bird, (0,0), fx=0.5, fy=0.5)

    two_small = np.concatenate((p_small, b_small), axis=0)
    return np.concatenate((rgb_lines, two_small), axis=1)

white_clip = clip1.fl_image(step)
%time white_clip.write_videofile(output, audio=False)
