#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

cv_image = None

def image_callback(msg):
    global cv_image

    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    except Exception as e:
        rospy.logerr(e)


def main() :

    rospy.init_node('image_display_node')
    rospy.Subscriber("/lidar_convert_image", Image, image_callback)
    
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if cv_image is not None:

            frame = cv_image
            frame = cv2.resize(frame, (640,480))

            # 여기서 작성






            cv2.imshow('Lidar Image', frame)

        if cv2.waitKey(1) & 0xFF == 27 :
            break


    cv2.destroyAllWindows()  # Close OpenCV window when done


if __name__ == '__main__':
    main()