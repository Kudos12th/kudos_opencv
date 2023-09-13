#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Global variable to hold the latest image
cv_image = None

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

def bev(image):
    # 사용 X
    transformed_img = image  
    return transformed_img

ye_low_bgr = np.array([0, 80, 80])
ye_upp_bgr = np.array([50, 150, 150])
detected_huddle_color = (255, 100, 255)

def calculate_distance(contour_area):
    
    
    return contour_area 

def main():
    rospy.init_node('video_subscriber_node', anonymous=True)

    # Subscribe to the "/kubot_cam/image_raw" topic
    rospy.Subscriber("/kubot_cam/image_raw", Image, image_callback)

    # Set the rate at which you want to display images (e.g., 10 Hz)
    rate = rospy.Rate(10)  # 10 Hz

    print("Press 'esc' to quit.")
    while not rospy.is_shutdown():
        
        if cv_image is not None:
            frame = cv_image
            frame = cv2.resize(frame, (640, 480))
            bev_frame = bev(frame)
    
            huddle_img = cv2.inRange(bev_frame, ye_low_bgr, ye_upp_bgr)
            huddle_contours, huddle_hier = cv2.findContours(huddle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            huddle_img = cv2.morphologyEx(huddle_img, cv2.MORPH_OPEN, kernel, iterations=2)

            huddle_contours, huddle_hier = cv2.findContours(huddle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = max(huddle_contours, key=cv2.contourArea, default=None)

            if largest_contour is not None:
                contour_area = cv2.contourArea(largest_contour)
                estimated_distance = calculate_distance(contour_area)
                print(f"Largest Contour Area: {contour_area}, Estimated Distance: {estimated_distance}")

            centers = []  

            for contour in huddle_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append((cX, cY))  

          
            sorted_centers = sorted(centers, key=lambda x: x[1], reverse=True)

    
            for idx, center in enumerate(sorted_centers, start=1):
                cX, cY = center
                print(f"Huddle {idx}: X = {cX}, Y = {cY}")
                cv2.circle(bev_frame, (cX, cY), 5, (0, 255, 0), -1)
            cv2.drawContours(bev_frame, huddle_contours, -1, detected_huddle_color , 2 ,cv2.LINE_8, huddle_hier)
            cv2.imshow("Detection in Bird Eye View", bev_frame)
            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
