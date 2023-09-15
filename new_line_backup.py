#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import math


cv_image = None
last_angular_vel = 0.0
last_linear_vel = 0.0


def image_callback(msg):
    global cv_image

    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    except Exception as e:
        rospy.logerr(e)



def bev(image):
    width=640
    height=480
    view=[[240, 300], [190, 480], [400, 300], [450, 480]] #좌상 좌하 우상 우하

    source = np.float32(view)
    destination = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (width, height))

    return transformed_img



def find_white(img):
    wh_low_bgr = np.array([100, 100, 100])
    wh_upp_bgr = np.array([255, 255, 255])

    white_img = cv2.inRange(img, wh_low_bgr, wh_upp_bgr)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_img = cv2.morphologyEx(white_img,cv2.MORPH_OPEN, kernel, iterations=2)
    line_contours, line_hier = cv2.findContours(white_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return line_contours, line_hier



def largest_center(line_contours, line_hier, out_img):
    print("Follow largest_contour")
    cv2.drawContours(out_img, line_contours, -1, [ 0, 0, 255], 2, cv2.LINE_8, line_hier)
    if line_contours:
        largest_line_contour = max(line_contours, key=cv2.contourArea)
        dst_point = tuple(largest_line_contour[0][0])
        cv2.circle(out_img, dst_point, 5, (0, 255, 0), -1)
    cv2.imshow("Detection in Bird Eye View", out_img)
    return dst_point



def top_center(line_contours, line_hier, out_img) :
    print("Follow top_contour")
    cv2.drawContours(out_img, line_contours, -1, [ 0, 0, 255], 2, cv2.LINE_8, line_hier)
    
    if line_contours:
        dst_point = tuple(line_contours[0][0][0])
        cv2.circle(out_img, dst_point, 5, (0, 255, 0), -1)
    cv2.imshow("Detection in Bird Eye View", out_img)

    return dst_point



def cal_vel(img_width, dst_x):
    global last_angular_vel, last_linear_vel

    center_x = img_width // 2

    if dst_x is not None:
        angle_error = math.atan2(center_x - dst_x, img_width) * 2.0

        max_angular_vel = 0.15
        max_linear_vel = 0.05

        angular_vel = max_angular_vel * angle_error
        linear_vel = max_linear_vel * (1.0 - abs(angle_error))

        last_angular_vel = angular_vel
        last_linear_vel = linear_vel

        return angular_vel, linear_vel
    else:
        # print("No line")
        return last_angular_vel, last_linear_vel


    
def main() :
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    rospy.init_node('video_subscriber_node', anonymous=True)
    rospy.Subscriber("/kubot_cam/image_raw", Image, image_callback)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if cv_image is not None :

            frame = cv_image
            frame = cv2.resize(frame, (640,480))

            bev_img = bev(frame)
            line_contours, line_hier = find_white(bev_img)
            dst_point = top_center(line_contours, line_hier, bev_img)
            angular_vel, linear_vel = cal_vel(bev_img.shape[1], dst_point[0])

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



        
