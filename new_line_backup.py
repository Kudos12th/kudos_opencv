#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import math

cv_image = None

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
    view=[[240, 100], [190, 480], [400, 100], [450, 480]] #좌상 좌하 우상 우하

    source = np.float32(view)
    destination = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (width, height))

    return transformed_img

def find_white(img, out_img):
    wh_low_bgr = np.array([100, 100, 100])
    wh_upp_bgr = np.array([255, 255, 255])

    detected_line_color = (0, 0, 255) 

    white_img = cv2.inRange(img, wh_low_bgr, wh_upp_bgr)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_img = cv2.morphologyEx(white_img,cv2.MORPH_OPEN, kernel, interations=2)
    line_contours, line_hier = cv2.findContours(white_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, line_contours, -1, detected_line_color, 2, cv2.LINE_8, line_hier)

    return out_img, line_contours

def cal_vel(line_contours):
    global last_angular_vel, last_linear_vel
    if line_contours:
        largest_line_contour = max(line_contours, key=cv2.contourArea)