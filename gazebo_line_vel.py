#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import math

# Global variable to hold the latest image
cv_image = None

def image_callback(msg):
    global cv_image

    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # You can also process the image data here as needed
        # ...

    except Exception as e:
        rospy.logerr(e)

# Bird's eye view transformation
def bev(image):
    # height, width = image.shape[:2]
    width = 640
    height = 480
    
    source = np.float32([[280, 350], [270, 430], [360, 350], [370, 430]])
    destination = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (640, 480))

    return transformed_img

wh_low_bgr = np.array([100, 100, 100])
wh_upp_bgr = np.array([255, 255, 255])

detected_line_color = (255, 150, 0)  # Light blue

# Calculate midpoints of the detected line
def calculate_midpoints(contour):
    midpoints = []
    for i in range(len(contour) - 1):
        x1, y1 = contour[i][0]
        x2, y2 = contour[i + 1][0]
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        midpoints.append((mid_x, mid_y))
    return midpoints

# Calculate angular and linear velocities based on detected line
def calculate_velocities(midpoints, image_width):
    if len(midpoints) >= 2:
        center_x = image_width // 2
        line_center_x = (midpoints[0][0] + midpoints[-1][0]) // 2
        angle_error = math.atan2(center_x - line_center_x, image_width) * 2.0
        
        max_angular_vel = 0.5
        max_linear_vel = 0.003
        
        angular_vel = max_angular_vel * angle_error
        linear_vel = max_linear_vel * (1.0 - abs(angle_error))
        
        return angular_vel, linear_vel
    else:
        return 0.0, 0.0

def main():
    rospy.init_node('video_subscriber_node', anonymous=True)
    rospy.Subscriber("/kubot_cam/image_raw", Image, image_callback)
    rate = rospy.Rate(10)

    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    print("Press 'esc' to quit.")
    while not rospy.is_shutdown():
        if cv_image is not None:
            frame = cv_image
            frame = cv2.resize(frame, (640, 480))
            bev_frame = bev(frame)
            line_img = cv2.inRange(bev_frame, wh_low_bgr, wh_upp_bgr)
            line_contours, line_hier = cv2.findContours(line_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            line_img = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, kernel, iterations=2)
            line_contours, line_hier = cv2.findContours(line_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if line_contours:
                line_largest_contour = max(line_contours, key=cv2.contourArea)
                midpoints = calculate_midpoints(line_largest_contour)
                
                angular_vel, linear_vel = calculate_velocities(midpoints, bev_frame.shape[1])

                # Publish velocities using cmd_vel topic
                twist = Twist()
                twist.angular.z = angular_vel
                twist.linear.x = linear_vel
                cmd_vel_pub.publish(twist)

                # Draw the detected line and midpoints
                for midpoint in midpoints:
                    cv2.circle(bev_frame, midpoint, 3, (0, 255, 0), -1)

            cv2.drawContours(bev_frame, line_contours, -1, detected_line_color, 2, cv2.LINE_8, line_hier)
    
            cv2.imshow("Camera", frame)
            cv2.imshow("Detection in Bird Eye View", bev_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
