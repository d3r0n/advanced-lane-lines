import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


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

        left_detection = LaneDetection(img, left_fit, leftx, lefty)
        right_detection = LaneDetection(img, right_fit, rightx, righty)

        return (left_detection, right_detection)

    def visualisation_elements(self, img):
        # making the original road pixels 3 color channels
        gr_img = img * 255
        self.visualisation = np.array(cv2.merge([gr_img,gr_img,gr_img]),np.uint8)
        left_detection, right_detection  = self.find_lane_polynomials(img)
        left_fit, right_fit = left_detection.fit, right_detection.fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return (self.visualisation, left_fitx, right_fitx, ploty)


class ConvolutionSearch:
    """WIP: needs refactoring but in general works, not great though"""

    def __init__(self, window_width = 50, window_height = 80, margin = 100):
        self.window_width = window_width
        self.window_height = window_height # Break image into 9 vertical layers since image height is 720
        self.margin = margin # How much to slide left and right for searching


    def find_window_centroids(self, image):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        warped = image

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
    	    # convolve the window into the vertical slice of the image
    	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
    	    conv_signal = np.convolve(window, image_layer)
    	    # Find the best left centroid by using past left center as a reference
    	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    	    offset = window_width/2
    	    l_min_index = int(max(l_center+offset-margin,0))
    	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
    	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    	    # Find the best right centroid by using past right center as a reference
    	    r_min_index = int(max(r_center+offset-margin,0))
    	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
    	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    	    # Add what we found for that layer
    	    window_centroids.append((l_center,r_center))

        return window_centroids

    def plot_visualisation(self, img):
        def window_mask(width, height, img_ref, center,level):
            output = np.zeros_like(img_ref)
            output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
            return output

        window_centroids = self.find_window_centroids(img)

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(convolution.window_width,convolution.window_height,img,window_centroids[level][0],level)
            r_mask = window_mask(convolution.window_width,convolution.window_height,img,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge([zero_channel,template,zero_channel]),np.uint8) # make window pixels green
        img = img * 255
        warpage = np.array(cv2.merge([img,img,img]),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(src1 = warpage, alpha=0.7, src2=template, beta=0.9, gamma=0.0) # overlay the orignal road image with window results

        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show()
