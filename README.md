[//]: # (Image References)
[image0]: ./images/header.gif "Header"
[image1]: ./images/distorted.jpg "Original"
[image2]: ./images/undistorted.jpg "Undistorted"
[image3]: ./images/distorted_test.jpg "Original test image"
[image4]: ./images/undistorted_test.jpg "Undistorted test image"
[image5]: ./images/threshold.jpg "Threshold"
[image6]: ./images/bird_eye.jpg "Bird-eye perspective"
[image7]: ./images/warped.jpg "Warped"
[image8]: ./images/histogram.png "Histogram"
[image8.1]: ./images/fit1.png "fit"
[image8.2]: ./images/fit2.jpg "fit"
[image8.3]: ./images/fit3.png "fit"
[image9.1]: ./images/curvature_eq.png "Radius of curvature equation"
[image9.2]: ./images/curvature_der.png "Second and first derivative of second order polynomial"
[image9.3]: ./images/curvature_eq_fin.png "Radius of curvature for second order polynomial"
[image9.4]: ./images/curvature.png "Curvature"
[image10.1]: ./images/out_1.gif "Output"
[image10.2]: ./images/out_2.gif "Output"
[image10.3]: ./images/out_3.gif "Output"

# Advanced Lane Finding Project

![alt text][image0]

>__TLDR;__ here is a link to my [main code.](https://github.com/d3r0n/advanced-lane-lines/blob/master/main.py)

---

### My achievements :rocket: in this project:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### Camera Calibration

#### 1. Calibration matrix and distortion coefficients computation

The code for this step in contained in file [camera.](https://github.com/d3r0n/advanced-lane-lines/blob/master/camera.py)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

__before__

![alt text][image1]

__after__

![alt text][image2]

---

### Pipeline

#### 1. Undistort

Before we start detecting anything we need to correct camera distortion. So applying coefficients from calibration is a first step.  

__before__

![alt text][image3]

__after__

![alt text][image4]

#### 2. Blur

In order to get more continuous lines I blur the image a little.  

#### 3. Threshold

Method `threshold` from `LaneLines` shows all used thresholds in pipeline. But If you are interested in implementations have a look at class Threshold from [threshod.py](https://github.com/d3r0n/advanced-lane-lines/blob/master/threshod.py)

I used a combination of color and gradient thresholds to generate a binary image.

>__Most valuable was to threshold converted image to different color representations.__

I have focused on two targets. First to detect white line and second to detect yellow line.
So in `LaneLines` class you can find combination of different thresholds specialized on detecting one of the mentioned. There was lot of gain changing color representations like using channel L from LUV for white line or channel B from LAB for yellow line. Surprisingly, I still keep threshold for over 200 pixels in RGB due to its good performance on white line detection. I found that basically each color channel used for detection like L from LUV can then be threshold with gradients. Either in respective to x axis, both x and y, or direction of the gradient. Using y axis threshold did not result in improvement of the detection or even made things worse. For instance when other car drives near the lane. It can be easily mistaken and the whole lane will be incorrect.

Here's an example of successfully threshold input image.
![alt text][image5]

#### 4. Perspective transform

Next step is to perform a perspective transformation in order to have get better view on the street.

__"bird-eye" perspective__

![alt text][image6]

If you are interested how I did it have a look at file
[perspective.py](https://github.com/d3r0n/advanced-lane-lines/blob/master/perspective.py)
This is just a wrapper around `cv2` method `warpPerspective`. The source and destination points are hardcoded in `Laneline` class like this:

```python
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
```
This results in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 565, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 715, 460      | 960, 0        |

Here, verify yourself on below example. For your convenience I  have drawn source points.
Test image on the right and its warped counterpart on the left.

![alt text][image7]

#### 5. Histogram search

After applying calibration, thresholding, and a perspective transform to a road image, the result binary image have the lane lines which stand out clearly.

If you do histogram of that binary image you will have something like this:
![alt text][image8]

See that peaks? It is where our lanes are most likely. These are the starting points of the histogram search. If you want see the code you can find it in  [lane_search.py](https://github.com/d3r0n/advanced-lane-lines/blob/master/lane_search.py)

The general idea is to start from points pointed by 2 biggest peaks in histogram search. One for left lane and one for right lane. When starting from bottom the algorithm searches for pixels near bottom. When found it puts box around them. In my case I choose `180px x 80px` box. In next step it searches in the area near top of previous box.
When found all pixels tries to fit best the next box. It is like building a block tower. If next block will be way of from below one the tower will fall or in our case we will detect something else than lane line. When we have all the pixels from all the left and right boxes. We just fit a second order polynomial and voila! lane polynomial :tada: :shipit:

![alt text][image8.1]
![alt text][image8.2]
![alt text][image8.3]

#### Bonus. Radius of curvature.

How I am sure that my polynomial are correct?

Best is to check if they don't break the rules. Since the videos has been captured in US then captured highways must follow radius of curvature restrictions in [US.](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC)

Here is the equation for the radius of the curvature:

![alt text][image9.1]

Since I fit second order polynomial to the found pixels it is easy to compute its first and second derivative:

![alt text][image9.2]

When applied to radius of the curvature equation:

![alt text][image9.3]

Code for it is quite easy and can be found in `LaneStatistics`

>__Remember to translate poly from pixel space to real world space before calculating radius of curvature!__

Additional statistic we can compute is how off from the centre of the road is the car. So after translating to real world meters from pixel space we just take absolute difference of left and right lane positions at the bottom of the screen.

Result:

![alt text][image9.4]

---

### Video

#### 1. Butter smooth

To make butter smooth transitions and reduce wobbling in the video use Exponential Smoothing.
So each new detected polynomial will be averaged whit previously detected lines with respect to α factor.

```python
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
```

#### 2. Results
![alt text][image10.1]

![alt text][image10.2]

![alt text][image10.3]

#### 3. Full video
Here's a [link to full video result](https://github.com/d3r0n/advanced-lane-lines/blob/master/project_output.mp4)

---

### TO DO:

#### 1. Yellow line is still not perfect in less sunny roads.
#### 2. Think what will happen when used other vehicle. The road might be off the image centre.
#### 3. What if started to rain, snow? Tunnel might interesting...
#### 4. Finding thresholds can be a tedious work. Automate with some classifier?
#### 5. Improve the line search in warped space with another classifier?
