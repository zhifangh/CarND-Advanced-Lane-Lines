## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the calibrate_camercode function of the IPython notebook located in "./EP2.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

The flowing is origal image

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\camera_cal\calibration3.jpg)

The flowing is undistort image.

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\img_undistort.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\straight_lines2_undistort.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step.  

The flowing is the result apply abs_sobel_thresh:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\img_bin_sobel.png)

The flowing is the result apply mag_thresh:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\img_bin_mag.png)

The flowing is the result apply dir_threshold:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\img_bin_dir.png)

The flowing is the result apply hls_select:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\hls_select.png)

The flowing is th last result used a combination of a few thresholds to generate a binary image:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\combined.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
s1 = [180,719]
s2 = [555, 470]
s3 = [725, 470]
s4 = [1130,719]

d1 = [280,719]
d2 = [280,  0]
d3 = [1000,  0]
d4 = [1000, 719]

src = np.array([s1,s2,s3,s4],dtype = "float32")
dst = np.array([d1,d2,d3,d4],dtype = "float32")
```

This resulted in the following source and destination points:

|  Source  | Destination |
| :------: | :---------: |
| 180,719  |   280,719   |
| 555, 470 |   280,  0   |
| 725, 470 |  1000,  0   |
| 1130,719 |  1000, 719  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The flowing is undistorted image with source points drawn:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\img_cpy.png)

The floinwg is warped result with dest points drawn:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\img_warper.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\search_around_poly.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated radius of curvature of the lane in the function：my_measure_curvature_real

I calculated position of the vehicle in the function: my_measure_vichle_position

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function my_pipline()`.  Here is an example of my result on a test image:

![](D:\hzf\udacity\project\CarND-Advanced-Lane-Lines\output_images\my_pipline.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/P2.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part is select src and dest points to calculate perspective transform matrix. A reasonable result is that a rectangle still was rectangle in warped result. 
