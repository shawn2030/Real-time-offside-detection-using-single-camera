"""
Name: Shounak Desai (sd9649@rit.edu)
Project Name - Real-time Offside Detection system using a Single Camera
This program consists of a class LineDetector which takes in continuous 
frames from an input video and detects the light and dark green strips on
the soccer pitch and draws the lines on the pitch in every input frame,
calculates the vanishing point and draw parallel lines passing via
all players on the field.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pose import PoseEstimator


class LineDetector:
    def __init__(self, frame) -> None:
        self.image = frame
        self.detected_edges = None


    def plot_pixel_analysis(self, segmented_img):
        """
        This method helps to plot the pixel analysis for every channel in blue,
        green and red channels from the input frame. I used the same method to plot
        the pixel analysis for different image spaces.
        segmented_img: segmented form of the input image        
        """
        
        # Split the image into its three color channels: Blue, Green, Red
        blue_channel, green_channel, red_channel = cv.split(segmented_img)

        # Calculate histogram for each color channel
        blue_hist = cv.calcHist([blue_channel], [0], None, [256], [0, 256])
        green_hist = cv.calcHist([green_channel], [0], None, [256], [0, 256])
        red_hist = cv.calcHist([red_channel], [0], None, [256], [0, 256])

        # Plot histograms
        plt.figure(figsize=(8, 6))
        plt.plot(blue_hist, color='blue', label='Blue')
        plt.plot(green_hist, color='green', label='Green')
        plt.plot(red_hist, color='red', label='Red')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Pixel Analysis')
        plt.legend()
        plt.savefig('plot/pixel_analysis.png')
  

    def segment_playing_area(self):
        """
        This method segments the playing area which only keeps the part of the image 
        where pixel values fall inside the lower green and higher green range (which was
        calculated using the plot_pixel_analysis method) which helps to keep only the soccer
        pitch which is required for further processing.
        """

        hsv_image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([20, 30, 20])
        upper_green = np.array([50, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv.inRange(hsv_image, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create a black image
        black_image = np.zeros_like(self.image)

        # Draw contours on the black image
        cv.drawContours(black_image, contours, -1, (255, 255, 255), thickness=cv.FILLED)

        # Mask the original image with the contours
        result_image = cv.bitwise_and(self.image, black_image)
       
        # Display the result
        # cv.imwrite("output/03_segment.png", result_image)

        return result_image


    def compute_vanishing_point(self, all_lines):
        intersection_points = []

        # print(all_lines)

        for index_lines1 in range(len(all_lines)):
            for index_lines2 in range(len(all_lines)):

                    if index_lines1 is not index_lines2:

                        x1, y1, x2, y2 = all_lines[index_lines1]
                        x3, y3, x4, y4 = all_lines[index_lines2]
                        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if denominator != 0:
                            intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                            intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
                            intersection_points.append((intersection_x, intersection_y))

        intersection_points = np.array(intersection_points)
        # print(intersection_points)

        # RANSAC parameters
        iterations = 1000
        threshold_distance = 100
        inliers_ratio = .3

        best_model = None
        best_inliers = []
        max_inliers = 0

        for _ in range(iterations):
            # Randomly sample points to form a model (line)
            sample_indices = np.random.choice(len(intersection_points), 2, replace=False)
            sampled_points = intersection_points[sample_indices]
            
            # Fit a line through the sampled points
            line_model = np.polyfit(sampled_points[:, 0], sampled_points[:, 1], deg=1)
            
            # Calculate distances from all points to the line
            distances = np.abs(np.polyval(line_model, intersection_points[:, 0]) - intersection_points[:, 1])
            
            # Count inliers (points within threshold distance)
            inliers = distances < threshold_distance
            num_inliers = np.sum(inliers)
            
            # Update best model if current model has more inliers
            if num_inliers > max_inliers:
                best_model = line_model
                best_inliers = intersection_points[inliers]
                max_inliers = num_inliers
                
            # Check for early termination if enough inliers found
            if num_inliers > len(intersection_points) * inliers_ratio:
                break

        vanishing_x = np.mean(best_inliers[:, 0])
        vanishing_y = np.mean(best_inliers[:, 1])
        vanishing_point = (vanishing_x, vanishing_y)

        return vanishing_point

    def plot_vanishing_point(self, vanishing_point):
        """
        This method draws or plots the calculated vanishing point and save plot in
        the plots directory. 

        vanishing_point: calculated vanishing point 
        """
        hough_lines_image = cv.imread('output/03_edges.png')
        vanishing_x, vanishing_y= vanishing_point

        plt.imshow(cv.cvtColor(hough_lines_image, cv.COLOR_BGR2RGB))
        plt.plot(vanishing_x, vanishing_y, 'ro')  # Plot vanishing point as a red dot
        plt.title("Image with Vanishing Point")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('plots/vanishing_point.png')

        # print("Vanishing Point:", vanishing_point)

        return vanishing_point


    def detect_offside_lines(self, segmented_image):
        """
        Perform hough transform on the segemented image to detect the edges 
        on the soccer pitch between dark and light green stips.
        segmented_image: frame with segmented playing area

        return:: preferred_coordinates: set of lines which are useful for calculating 
                                    the vanishing point (x1, y1, x2, y2)
        """
        
        hsv_image = cv.cvtColor(segmented_image, cv.COLOR_BGR2HSV)
        green_channel = hsv_image[:, :, 1]  
        # cv.imwrite("output/03_green.png", green_channel)

        edges = cv.Canny(green_channel, threshold1=0, threshold2=40)  # Adjust thresholds as needed

        # cv.imwrite('output/edges.png', edges)

        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=230, minLineLength=300, maxLineGap=50)
        slope_list = []
     
        # Draw detected lines on the original image
        all_lines = []
        preferred_co_ordinates = [[0,0,0,0],[0,0,0,0]]
        preferred_co_ordinates_2 = [[0,0,0,0],[0,0,0,0]]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = x2 - x1
                dy = y2 - y1
                angle_rad = np.arctan2(dy, dx)
                angle_degree = np.degrees(angle_rad)

                if np.abs(angle_degree) > 18  and np.abs(angle_degree) < 45 and round(angle_degree) not in slope_list and angle_degree > 0:
                    # print(angle_degree)
                    slope_list.append(round(angle_degree))
                    cv.line(segmented_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    all_lines.append([x1, y1, x2, y2])

        # cv.imwrite("output/03_edges.png", image_copy)
        # print(slope_list)
        return segmented_image, lines, all_lines, preferred_co_ordinates
    
    def draw_detected_lines(self, preferred_co_ordinates):
        """
        This method draws the detected lines (strips) on the soccer pitch.
        preferred_co_ordinates: detected lines (x1, y1, x2, y2)

        return:: image_copy: detected lines drawn image
        """
        # print(preferred_co_ordinates)
        image_copy = self.image.copy()
        for co_ordinates in preferred_co_ordinates:
            x1, y1, x2, y2 = co_ordinates
            cv.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 4)

        # cv.imwrite("output/detected_lines.png", image_copy)
        return image_copy

    def draw_pose_lines(self, vanishing_point, image):
        """
        Draw parallel lines (in 3D space) from vanishing point on the input image and
        keypose collection of points.
        Vanishing_point: calculated vanishing point
        image: original input frame

        return:: img_out: pose detected image
        """
        model = PoseEstimator(0.9, 0.2)
        img_out, pose_points = model.detect_and_draw(image)
        # print(pose_points)
        vpx = vanishing_point[0]
        vpy = vanishing_point[1]

        pose_points = [(x2, y2) for x2, y2 in pose_points]


        for point in pose_points:
            # print(point)
            x, y = point[0], point[1]

            cv.line(img_out, (int(vpx), int(vpy)), (x, y), (0, 255, 255), 2)  # Red color, line thickness 2

        # Display or save the image with lines drawn
        # cv.imwrite("output/pose_lines.png", img_out)

        return img_out



if __name__ == "__main__":
    """
    Testing the LineDetector Object.
    """
    line_detector = LineDetector('dataset/01.png')
    segmented_img = line_detector.segment_playing_area()
    _, lines, all_lines, preferred_lines = line_detector.detect_offside_lines(segmented_img)
    vanishing_point = line_detector.compute_vanishing_point(all_lines)
    line_detector.draw_detected_lines(all_lines)
    line_detector.draw_pose_lines(vanishing_point)

