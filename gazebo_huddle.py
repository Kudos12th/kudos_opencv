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

def bev(image) :

    # height, width = image.shape[:2]

    # ource = np.float32([[width/3, 0], [0, height], [width*2/3, 0], [width, height]])
    # destination = np.float32([[0,0], [0,height],[width,0] ,[width, height] ])

    # transform_matrix = cv2.getPerspectiveTransform(source, destination)
    # transformed_img = cv2.warpPerspective(image, transform_matrix, (width,height))

    transformed_img = image  # 버드아이뷰 안쓸 때

    return transformed_img

 # yellow_bgr = [60,255,255]
ye_low_bgr = np.array([0, 80, 80])
ye_upp_bgr = np.array([50, 150, 150])

detected_huddle_color = (255 , 100 , 255) # light violet



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

            frame = cv2.resize(frame, (640,480))

            bev_frame = bev(frame)
    
            huddle_img = cv2.inRange(bev_frame, ye_low_bgr, ye_upp_bgr)

            huddle_contours, huddle_hier = cv2.findContours(huddle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 노이즈 제거를 위한 모폴로지 연산 적용
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            huddle_img = cv2.morphologyEx(huddle_img, cv2.MORPH_OPEN, kernel, iterations=2)

            huddle_contours, huddle_hier = cv2.findContours(huddle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(bev_frame, huddle_contours, -1, detected_huddle_color , 2 ,cv2.LINE_8, huddle_hier)
            cv2.imshow("Detection in Bird Eye View", bev_frame)
            cv2.imshow("Camera", frame)


        if cv2.waitKey(1) & 0xFF == 27 :
            break


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()