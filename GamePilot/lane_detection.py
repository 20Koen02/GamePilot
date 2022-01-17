import cv2
import numpy as np
from matplotlib import pyplot as plt

from GamePilot.gui import gui
from GamePilot.steering import steering

ENABLE_PLT = False

# Global variables
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []


class LaneDetection:
    def __init__(self, capture, docs_plot=False):
        self.docs_plot = docs_plot

        resized_capture = cv2.resize(capture, (640, 360))
        split_capture = resized_capture[int(resized_capture.shape[0] / 2):int(resized_capture.shape[0])]
        self.orig_frame = cv2.cvtColor(split_capture, cv2.COLOR_BGRA2BGR)

        self.orig_image_size = self.orig_frame.shape[::-1][1:]
        self.width = self.orig_image_size[0]
        self.height = self.orig_image_size[1]

        self.margin = self.width / 12
        self.minpix = self.width / 24
        self.no_of_windows = 10

        # this is a rough estimate
        self.YM_PER_PIX = 7.0 / 400  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 255  # meters per pixel in x dimension

        self.histogram = None
        self.warped = None
        self.preprocessed = None
        self.left_fit = None
        self.right_fit = None
        self.inv_transformation_matrix = None
        self.left_curvem = None
        self.right_curvem = None
        self.curve_radius = None
        self.center_offset = None

        self.perspective_transform()
        self.preprocessing()
        self.calculate_histogram()
        self.get_lane_line_indices_sliding_windows()
        self.get_lane_line_previous_window()
        self.overlay_lane_lines()
        self.calculate_curvature()
        self.calculate_car_position()
        steering.calc_angle(self.curve_radius, self.center_offset)

    def perspective_transform(self):
        orig_frame = self.orig_frame.copy()

        width = orig_frame.shape[1]
        height = orig_frame.shape[0]
        window_size = (width, height)

        src = np.float32([
            [(-gui.skew_bottom), height - gui.expand_bottom],
            [width + gui.skew_bottom, height - gui.expand_bottom],
            [width / 2 + gui.skew_top, (height / 2) + gui.expand_top],
            [width / 2 - gui.skew_top, (height / 2) + gui.expand_top]
        ])

        dst = np.float32(
            [[0, height],
             [width, height],
             [width, 0],
             [0, 0]])

        m = cv2.getPerspectiveTransform(src, dst)
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(dst, src)

        self.warped = cv2.warpPerspective(orig_frame, m, window_size, flags=cv2.INTER_LINEAR)

        if self.docs_plot:
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 3 rows, 1 column
            figure.tight_layout()
            ax1.imshow(self.orig_frame)
            ax2.imshow(cv2.resize(self.warped, (550, 450)))
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            plt.show()
        elif gui.skewed_view_enable:
            img = cv2.resize(self.warped, (550, 450))
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            gui.window["view_image"].update(data=imgbytes)

    def preprocessing(self):
        blur = cv2.GaussianBlur(self.warped, (15, 15), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        mask_white = cv2.inRange(gray, gui.white_tresh, 255)
        non_adaptive = cv2.bitwise_and(gray, mask_white)
        self.preprocessed = non_adaptive

        # Use adaptive threshold if less than 1% of the mask is white
        percentage_white = (self.preprocessed > 0).mean() * 100
        if percentage_white < 1 or self.docs_plot:
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,
                                                    -3)
            adaptive = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            if percentage_white < 1:
                self.preprocessed = adaptive

        if self.docs_plot:
            figure, (ax1, ax2, ax3) = plt.subplots(1, 3)  # 1 row, 3 columns
            figure.tight_layout()
            ax1.imshow(cv2.resize(self.warped, (550, 450)))
            ax2.imshow(cv2.resize(adaptive, (550, 450)), cmap='gray')
            ax3.imshow(cv2.resize(non_adaptive, (550, 450)), cmap='gray')
            ax1.set_title("Warped")
            ax2.set_title("Adaptive")
            ax3.set_title("Non-adaptive")
            plt.show()
        elif gui.preprocessing_view_enable:
            img = cv2.resize(self.preprocessed, (550, 450))
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            gui.window["view_image"].update(data=imgbytes)

    def calculate_histogram(self):
        self.histogram = np.sum(self.preprocessed[int(self.preprocessed.shape[0] / 2):, :], axis=0)

        if self.docs_plot:
            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(1, 2)  # 2 row, 1 columns
            figure.tight_layout()
            ax1.imshow(cv2.resize(self.preprocessed, (550, 450)), cmap='gray')
            ax1.set_title("Warped Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

    def get_lane_line_indices_sliding_windows(self, plot=False):
        # Sliding window width is +/- margin
        margin = self.margin

        frame_sliding_window = self.preprocessed.copy()

        # Set the height of the sliding windows
        window_height = np.int(self.preprocessed.shape[0] / self.no_of_windows)

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.preprocessed.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store the pixel indices for the left and right lane lines
        left_lane_inds = []
        right_lane_inds = []

        # Current positions for pixel indices for each window,
        # which we will continue to update
        midpoint = np.int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Go through one window at a time
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.preprocessed.shape[0] - (window + 1) * window_height
            win_y_high = self.preprocessed.shape[0] - window * window_height
            win_xleft_low = int(leftx_current - margin)
            win_xleft_high = int(leftx_current + margin)
            win_xright_low = int(rightx_current - margin)
            win_xright_high = int(rightx_current + margin)
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
                win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
                win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (
                                      nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (
                                       nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract the pixel coordinates for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial curve to the pixel coordinates for
        # the left and right lane lines
        left_fit = None
        right_fit = None

        global prev_leftx
        global prev_lefty
        global prev_rightx
        global prev_righty
        global prev_left_fit
        global prev_right_fit

        # Make sure we have nonzero pixels
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx
            lefty = prev_lefty
            rightx = prev_rightx
            righty = prev_righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Add the latest polynomial coefficients
        prev_left_fit.append(left_fit)
        prev_right_fit.append(right_fit)

        # Calculate the moving average
        if len(prev_left_fit) > 10:
            prev_left_fit.pop(0)
            prev_right_fit.pop(0)
            left_fit = sum(prev_left_fit) / len(prev_left_fit)
            right_fit = sum(prev_right_fit) / len(prev_right_fit)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx = leftx
        prev_lefty = lefty
        prev_rightx = rightx
        prev_righty = righty

        if self.docs_plot or gui.window_view_enable:
            # Create the x and y values to plot on the image
            ploty = np.linspace(
                0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # Generate an image to visualize the result
            out_img = np.dstack((
                frame_sliding_window, frame_sliding_window, (
                    frame_sliding_window))) * 255

            # Add color to the left line pixels and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]

            if self.docs_plot:
                # Plot the figure with the sliding windows
                figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
                figure.tight_layout()
                ax1.imshow(self.preprocessed)
                ax2.imshow(frame_sliding_window, cmap='gray')
                ax3.imshow(out_img)
                ax3.plot(left_fitx, ploty, color='yellow')
                ax3.plot(right_fitx, ploty, color='yellow')
                ax1.set_title("Warped Frame")
                ax2.set_title("Warped Frame with Sliding Windows")
                ax3.set_title("Detected Lane Lines with Sliding Windows")
                plt.show()
            elif gui.window_view_enable:
                img = cv2.resize(frame_sliding_window, (550, 450))
                imgbytes = cv2.imencode('.png', img)[1].tobytes()
                gui.window["view_image"].update(data=imgbytes)

        return self.left_fit, self.right_fit

    def get_lane_line_previous_window(self):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param: left_fit Polynomial function of the left lane line
        :param: right_fit Polynomial function of the right lane line
        :param: plot To display an image or not
        """

        left_fit = self.left_fit
        right_fit = self.right_fit

        # margin is a sliding window parameter
        margin = self.margin

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.preprocessed.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store left and right lane pixel indices
        left_lane_inds = ((nonzerox > (left_fit[0] * (
                nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                                  nonzerox < (left_fit[0] * (
                                  nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (
                nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                                   nonzerox < (right_fit[0] * (
                                   nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Get the left and right lane line pixel locations
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        global prev_leftx2
        global prev_lefty2
        global prev_rightx2
        global prev_righty2
        global prev_left_fit2
        global prev_right_fit2

        # Make sure we have nonzero pixels
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx2
            lefty = prev_lefty2
            rightx = prev_rightx2
            righty = prev_righty2

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Add the latest polynomial coefficients
        prev_left_fit2.append(left_fit)
        prev_right_fit2.append(right_fit)

        # Calculate the moving average
        if len(prev_left_fit2) > int(gui.moving_avg):
            prev_left_fit2.pop(0)
            prev_right_fit2.pop(0)
            left_fit = sum(prev_left_fit2) / len(prev_left_fit2)
            right_fit = sum(prev_right_fit2) / len(prev_right_fit2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx2 = leftx
        prev_lefty2 = lefty
        prev_rightx2 = rightx
        prev_righty2 = righty

        # Create the x and y values to plot on the image
        ploty = np.linspace(
            0, self.preprocessed.shape[0] - 1, self.preprocessed.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

    def overlay_lane_lines(self):
        """
        Overlay lane lines on the original frame
        :param: Plot the lane lines if True
        :return: Lane with overlay
        """
        # Generate an image to draw the lane lines on
        warp_zero = np.zeros_like(self.preprocessed).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([
            self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
            self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
            self.orig_frame.shape[
                1], self.orig_frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if self.docs_plot:
            figure, (ax1) = plt.subplots(1, 1)  # 3 rows, 1 column
            ax1.imshow(cv2.resize(result, (550, 450)))
            ax1.set_title("Lane Overlay")
            plt.show()
        elif gui.lines_view_enable:
            img = cv2.resize(result, (550, 450))
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            gui.window["view_image"].update(data=imgbytes)

        return result

    def calculate_curvature(self):
        # Set the y-value where we want to calculate the road
        # curvature.
        # Select the maximum y-value, which is the bottom of the frame.
        y_eval = np.max(self.ploty)

        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
            self.XM_PER_PIX), 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
            self.XM_PER_PIX), 2)

        # Calculate the radii of curvature
        left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[
            1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[
            0] * y_eval * self.YM_PER_PIX + right_fit_cr[
                                  1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        self.curve_radius = (self.left_curvem + self.right_curvem) / 2

        if not self.docs_plot:
            gui.window["curve_radius"].update(
                value='Curve Radius: ' + str((self.left_curvem + self.right_curvem) / 2)[:7])

    def calculate_car_position(self):
        # Assume the camera is centered in the image.
        # Get position of car in centimeters
        car_location = self.orig_frame.shape[1] / 2

        # Fine the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0] * height ** 2 + self.left_fit[
            1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height ** 2 + self.right_fit[
            1] * height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(
            center_lane)) * self.XM_PER_PIX * 100

        self.center_offset = center_offset

        if not self.docs_plot:
            gui.window["center_offset"].update(value='Center Offset: ' + str(self.center_offset)[:7] + ' cm')
