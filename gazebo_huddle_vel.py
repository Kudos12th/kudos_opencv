#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from geometry_msgs.msg import Twist


# Global variable to hold the latest image
cv_image = None
last_angular_vel = 0.0
last_linear_vel = 0.0

def image_callback(msg):

    global cv_image

    try:
        # Convert the ROS Image message to OpenCV format
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # You can also process the image data here as needed
        # ...

    except Exception as e:
        rospy.logerr(e)


def calculate_velocities(line_centers, huddle_centers, image_width):
    global last_angular_vel, last_linear_vel

    if huddle_centers and huddle_centers[-1][1] > 350 :
        return 0.0, 0.0

    elif len(line_centers) >= 2 :
        angular_vel = 0.0
        linear_vel = 0.0

        center_x = image_width // 2
        line_center_x = sum(point[0] for point in line_centers) // len(line_centers)
        angle_error = math.atan2(center_x - line_center_x, image_width) * 2.0

        max_angular_vel = 0.15
        max_linear_vel = 0.05

        angular_vel = max_angular_vel * angle_error
        linear_vel = max_linear_vel * (1.0 - abs(angle_error))

        last_angular_vel = angular_vel
        last_linear_vel = linear_vel
        
        return angular_vel, linear_vel

    else :
        return last_angular_vel, last_linear_vel
   


def find_white(image) :

    wh_low_bgr = np.array([100, 100 , 100])
    wh_upp_bgr = np.array([255, 255, 255])
    line_img = cv2.inRange(image, wh_low_bgr, wh_upp_bgr)

    return line_img


def find_yellow(image) :

    ye_low_bgr = np.array([0, 80, 80])
    ye_upp_bgr = np.array([50, 150, 150])
    huddle_img = cv2.inRange(image, ye_low_bgr, ye_upp_bgr)
    
    return huddle_img

# bird eye view
def bev(image) :

    # height, width = image.shape[:2]
    width = 640 #320
    height = 480 #240
    
    # source = np.float32([[280, 440], [270, 480], [360, 440], [370, 480]])
    source = np.float32([[200, 430], [190, 480], [440, 430], [450, 480]])
    destination = np.float32([[0,0], [0,height],[width,0] ,[width, height] ])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (640,480))

    # transformed_img = image  # 버드아이뷰 안쓸 때

    return transformed_img


def window(image,out_img) :

    margin = 50
    nwindows = 5
    window_height = int(image.shape[0]/nwindows)
    line_color = [255,0,0]
    window_color = [0, 0, 255]
    center_color = [0, 255, 0]
    thickness = 2
    line_centers = []

    line_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, line_contours, -1, line_color, thickness, cv2.LINE_8, _)

    for w in range(nwindows) :
        win_y_low = image.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = image.shape[0] - w * window_height # winodw 아랫부분

        points_in_window = []

        for contour in line_contours :
            for point in contour :
                if win_y_low <= point[0][1] <= win_y_high :
                    points_in_window.append(point[0])

        
        if len(points_in_window) > 0 :
            points_in_window = np.array(points_in_window)
            M = cv2.moments(points_in_window)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                line_centers.append((cX, cY)) 

                win_x_low = cX - margin
                win_x_high = cX + margin

                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), window_color, thickness)

    if len(line_centers) > 1:
        for i in range(len(line_centers) -1):
            cv2.line(out_img, line_centers[i], line_centers[i+1], center_color, thickness)

    cv2.imshow('detection in Bird Eye View', out_img)

    return line_centers


def huddle_detect(image,out_img) :
    huddle_color = (255, 100, 255)
    huddle_centers = []
    
    huddle_contours, huddle_hier = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # largest_contour = max(huddle_contours, key=cv2.contourArea, default=None)
    # if largest_contour is not None:
    #             contour_area = cv2.contourArea(largest_contour)
    #             # estimated_distance = calculate_distance(contour_area)
    #             print(f"Largest Contour Area: {contour_area}") 
        
    for contour in huddle_contours :
        M = cv2.moments(contour)
        if M["m00"] != 0 :
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            huddle_centers.append((cX, cY)) 

    sorted_centers = sorted(huddle_centers, key=lambda x: x[1], reverse=True)
    for idx, center in enumerate(sorted_centers, start=1):
                cX, cY = center
                print(f"Huddle {idx}: X = {cX}, Y = {cY}")
                cv2.circle(out_img, (cX, cY), 5, (0, 255, 0), -1)

    cv2.drawContours(out_img, huddle_contours, -1, huddle_color , 2 ,cv2.LINE_8, huddle_hier)
    cv2.imshow("Detection in Bird Eye View", out_img)

    return sorted_centers


def main():
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    rospy.init_node('video_subscriber_node', anonymous=True)

    # Subscribe to the "/kubot_cam/image_raw" topic
    rospy.Subscriber("/kubot_cam/image_raw", Image, image_callback)

    # Set the rate at which you want to display images (e.g., 10 Hz)
    rate = rospy.Rate(10)  # 10 Hz

    print("Press 'esc' to quit.")
    while not rospy.is_shutdown():
        # Display the image using OpenCV
        if cv_image is not None:

            frame = cv_image
            frame = cv2.resize(frame, (640, 480))
            bev_img = bev(frame)

            wh_img = find_white(bev_img)
            line_centers = window(wh_img, bev_img)

            ye_img = find_yellow(frame)
            huddle_centers = huddle_detect(ye_img, frame)

            # Calculate angular and linear velocities using centers
            angular_vel, linear_vel = calculate_velocities(line_centers,huddle_centers ,bev_img.shape[1])

            # Publish velocities using cmd_vel topic
            twist = Twist()
            twist.angular.z = angular_vel
            twist.linear.x = linear_vel
            cmd_vel_pub.publish(twist)

            cv2.imshow('camera', frame)


        if cv2.waitKey(1) & 0xFF == 27 :
            break


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()