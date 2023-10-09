import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist

rospy.init_node('ball_walk')
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

cap = cv2.VideoCapture(3)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Frame size: ", frame_size)

def find_blue(img):

    # bgr
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 150, 50])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    white_img = cv2.inRange(img, lower_blue, upper_blue)

    blue_contours, blue_hier = cv2.findContours(white_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # hsv
    # hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_blue_hsv = np.array([100, 50, 50])
    # upper_blue_hsv = np.array([130, 255, 255])
    # blue_img = cv2.inRange(hsv_image, lower_blue_hsv, upper_blue_hsv)

    # blue_contours, blue_hier = cv2.findContours(blue_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return blue_contours, blue_hier

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    blue_contours, blue_hier = find_blue(frame)
    cv2.drawContours(frame, blue_contours, -1, (0, 0, 255), 2, cv2.LINE_8, blue_hier)

    if blue_contours:
        blue_y = [point[0][1] for contour in blue_contours for point in contour]
        max_blue_y = max(blue_y)
        cv2.line(frame, (0,max_blue_y), (640,max_blue_y), (255,0,0), (2))
        if max_blue_y >= 240:
            angular_vel, linear_vel = 0.0, 0.0
            print('max_blue_y = ', max_blue_y,'\n STOP')

        else:
            angular_vel, linear_vel = 0.0, 0.4
            print('max_blue_y = ', max_blue_y)
    else:
        angular_vel, linear_vel = 0.0, 0.4
        print('No Blue')

    cv2.imshow("Camera", frame)

    twist = Twist()
    twist.angular.z = angular_vel
    twist.linear.x = linear_vel
    cmd_vel_pub.publish(twist)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
