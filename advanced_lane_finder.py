# All imports defined here
# complete code is in this cell
import numpy as np
import cv2
import glob
import os
import random
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


class CameraCalibrator(object):

    def __init__(self, camera_calibration_image_list, nx, ny):
        self.camera_calibration_images = camera_calibration_image_list
        self.chessboard_return = []
        self.chessboard_corners = []
        self.nx = nx
        self.ny = ny
        self.valid_index = []
        self.mtx = None
        self.dist = None
        
    
    def calibrate(self):
        imgpoints_list = []
        objpoints_list = []
        objpoints = np.zeros((self.nx*self.ny, 3), np.float32)
        objpoints[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1, 2)
        for f_run, fname in enumerate(self.camera_calibration_images):
            gray = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny),None)
            self.chessboard_return.append(ret)
            self.chessboard_corners.append(corners)
            # If found, add object points, image points
            if ret == True:
                self.valid_index.append(f_run)
                objpoints_list.append(objpoints)
                imgpoints_list.append(corners)
        # calibrate the camera
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_list,
                                                                     imgpoints_list,
                                                                     gray.shape[::-1],
                                                                     None, None)

    def undistort_the_image(self, input_img):
        return cv2.undistort(input_img, self.mtx, self.dist, None, self.mtx)

    def get_chessboard_corners(self, image_index):
        return self.chessboard_return[image_index], self.chessboard_corners[image_index]
        
    def get_camera_calib_image_list(self):
        return self.camera_calibration_images

    def get_valid_indexes(self):
        return self.valid_index


class BirdViewTranformer(object):

    def __init__(self, input_image, src_points):
        self.warp_transform = None
        self.warp_inverse_transform = None
        self.src_points = src_points
        self.dst_points = []
        image_copy = np.copy(input_image)
        offset = 300
        image_size_y, image_size_x, _ =  input_image.shape
        src = np.float32(src_points)
        self.dst_points = np.array([[offset, 0],
                                    [offset, image_size_y],
                                    [image_size_x-offset, image_size_y],
                                    [image_size_x-offset, 0]], np.float32)
        self.warp_transform = cv2.getPerspectiveTransform(self.src_points, self.dst_points )
        self.warp_inverse_transform = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def change_to_birds_eye_view(self, non_warp_image):
        warp_image = cv2.warpPerspective(non_warp_image, self.warp_transform, 
                                        (non_warp_image.shape[1],
                                         non_warp_image.shape[0]),
                                         flags=cv2.INTER_LINEAR)
        return warp_image

    def change_back_from_birds_view(self, warp_image):
        non_warp_image = cv2.warpPerspective(warp_image, self.warp_inverse_transform, 
                                    (warp_image.shape[1],
                                     warp_image.shape[0]),
                                     flags=cv2.INTER_LINEAR)
        return non_warp_image

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,  ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,  ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(sobel_grad)
    binary_output[(sobel_grad >= thresh[0]) & (sobel_grad <= thresh[1])] = 1
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel_mag = np.sqrt((sobelx**2) + (sobely**2))
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def abs_sobel_thresh(gray, sobel_kernel=3, orient='x', thresh_min=0, thresh_max=255):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,  ksize=sobel_kernel)
    if orient == 'x':
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    elif orient == 'y':
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def pipeline(input_image):
    image_copy = np.copy(input_image)
    image_copy = cv2.GaussianBlur(image_copy,(3,3),0)
    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # x - vertical lines in the image
    # y - horizontal line in the image
    kernel_size = 5
    abs_sobelx_out = abs_sobel_thresh(s_channel, kernel_size, 'x', 20, 100)
    abs_sobely_out = abs_sobel_thresh(s_channel, kernel_size, 'y', 20, 100)
    mag_sobel = mag_thresh(s_channel, kernel_size, mag_thresh=(20, 100))
    dir_sobel = dir_threshold(s_channel, kernel_size, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_sobel)
    combined[((abs_sobelx_out == 1) & (abs_sobely_out == 1)) | ((mag_sobel == 1) & (dir_sobel == 1))] = 1
    morph_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, morph_struct, iterations=3)
    return combined
   
class LaneFinder(object):

    def __init__(self, camera_calibrator, bird_view_transformer):
        self.init_frame = True
        self.average_window_size = 10
        self.average_window_index = 0
        self.iteration_counter = 0
        self.left_lane_func = None
        self.right_lane_func = None
        self.left_fitx = []
        self.right_fitx = []
        self.left_fitx_array = np.zeros((self.average_window_size , 720))
        self.right_fitx_array = np.zeros((self.average_window_size , 720))
        self.camera_calibrator = camera_calibrator
        self.bird_view_transformer = bird_view_transformer
        self.lane_stats = []
    
    def make_image_undistorted(self,input_image):
        return self.camera_calibrator.undistort_the_image(input_image)
    
    def get_birds_eye_view(self, input_image):
        return self.bird_view_transformer.change_to_birds_eye_view(input_image)
    
    def get_normal_view(self, input_image):
        return self.bird_view_transformer.change_back_from_birds_view(input_image)
    
    def sanity_check(self, left_fitx, right_fitx):
        lanes_ok = False
        lane_distances = right_fitx - left_fitx
        lane_length_std_dev = np.std(lane_distances)
        #print ('lane distance std dev - ' , str(lane_length_std_dev))
        if lane_length_std_dev < 15:
            lanes_ok = True
        return lanes_ok
    
    def fit_polynomial_for_lanes_faster(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 50
        left_lane_inds = ((nonzerox > (self.left_lane_func[0]*(nonzeroy**2) + self.left_lane_func[1]*nonzeroy + self.left_lane_func[2] - margin)) & (nonzerox < (self.left_lane_func[0]*(nonzeroy**2) + self.left_lane_func[1]*nonzeroy + self.left_lane_func[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_lane_func[0]*(nonzeroy**2) + self.right_lane_func[1]*nonzeroy + self.right_lane_func[2] - margin)) & (nonzerox < (self.right_lane_func[0]*(nonzeroy**2) + self.right_lane_func[1]*nonzeroy + self.right_lane_func[2] + margin)))  
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_lane_func = np.polyfit(lefty, leftx, 2)
        right_lane_func = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_lane_func[0]*ploty**2 + left_lane_func[1]*ploty + left_lane_func[2]
        right_fitx = right_lane_func[0]*ploty**2 + right_lane_func[1]*ploty + right_lane_func[2]
        # check if the fitted function is valid
        if self.sanity_check(left_fitx, right_fitx):
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx
            self.left_lane_func = left_lane_func
            self.right_lane_func = right_lane_func
        else:
            self.init_frame = True
        
    def fit_polynomial_for_lanes(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        # Fit a second order polynomial to each
        left_lane_func = np.polyfit(lefty, leftx, 2)
        right_lane_func = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_lane_func[0]*ploty**2 + left_lane_func[1]*ploty + left_lane_func[2]
        right_fitx = right_lane_func[0]*ploty**2 + right_lane_func[1]*ploty + right_lane_func[2]
        if self.sanity_check(left_fitx, right_fitx):
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx
            self.left_lane_func = left_lane_func
            self.right_lane_func = right_lane_func
        else:
            self.init_frame = True

    def calculate_curvature(self, warped_image, left_fitx, right_fitx):
        ploty = np.linspace(0, warped_image.shape[0]-1, num=warped_image.shape[0])
        y_eval = np.max(ploty)
        left_lane_intercept = (self.left_lane_func[0]*y_eval)**2 + self.left_lane_func[1]*y_eval + self.left_lane_func[2]
        right_lane_intercept = (self.right_lane_func[0]*y_eval)**2 + self.right_lane_func[1]*y_eval + self.right_lane_func[2]
        lane_width = right_lane_intercept - left_lane_intercept
        ym_per_pix = 30.0/warped_image.shape[0] # meters per pixel in y dimension
        xm_per_pix = 3.7/lane_width # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        calculated_lane_center = (left_lane_intercept + right_lane_intercept) / 2.0
        lane_deviation = (calculated_lane_center - warped_image.shape[1] / 2.0) * xm_per_pix
        self.lane_stats = [left_curverad, right_curverad, lane_deviation]
 
    def overlay_detected_lane(self, warped_image, left_fitx, right_fitx):
        ploty = np.linspace(0, warped_image.shape[0]-1, num=warped_image.shape[0])
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        return self.get_normal_view(color_warp) 

    def process_each_frame(self, current_image):
        undist_img = self.make_image_undistorted(current_image)
        binary_image = pipeline(undist_img)    
        after_warp = self.get_birds_eye_view(binary_image)
        if self.init_frame:
            self.fit_polynomial_for_lanes(after_warp)
            self.init_frame = False
        else:
            self.fit_polynomial_for_lanes_faster(after_warp)

        self.left_fitx_array[self.average_window_index, :] = self.left_fitx
        self.right_fitx_array[self.average_window_index, :] = self.right_fitx
        self.average_window_index += 1
        if self.average_window_index >= self.average_window_size:
            self.average_window_index = 0

        if self.iteration_counter < self.average_window_size:
            self.iteration_counter += 1
            mean_left_fitx = np.sum(self.left_fitx_array, axis=0)/float(self.iteration_counter)
            mean_right_fitx = np.sum(self.right_fitx_array, axis=0)/float(self.iteration_counter)
        else:
            mean_left_fitx = np.mean(self.left_fitx_array, axis=0)
            mean_right_fitx = np.mean(self.right_fitx_array, axis=0)
    
        self.calculate_curvature(after_warp, mean_left_fitx, mean_right_fitx)
        detected_lane_image = self.overlay_detected_lane(after_warp, mean_left_fitx, mean_right_fitx)
        rad_curv = round((self.lane_stats[0] + self.lane_stats[1])/2.0, 2)
        lane_dev = round(self.lane_stats[2], 2)
        result = cv2.addWeighted(undist_img, 1, detected_lane_image, 0.3, 0)
        cv2.putText(result, 'Radius of Curvature = ' + str(rad_curv) + 'm',
                   (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (255,0,0), 2)
        cv2.putText(result, 'Lane-Deviation = ' + str(lane_dev) + 'm',
                   (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0,0,255), 2)
        return result
        
    
        
        
