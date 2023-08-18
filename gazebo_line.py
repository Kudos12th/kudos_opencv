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


# bird eye view
def bev(image) :

    # height, width = image.shape[:2]
    width = 640
    height = 480
    
    # source = np.float32([[280, 440], [270, 480], [360, 440], [370, 480]])
    source = np.float32([[280, 350], [270, 430], [360, 350], [370, 430]])
    destination = np.float32([[0,0], [0,height],[width,0] ,[width, height] ])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (640,480))

    # transformed_img = image  # 버드아이뷰 안쓸 때

    return transformed_img


wh_low_bgr = np.array([100, 100 , 100])
wh_upp_bgr = np.array([255, 255, 255])

detected_line_color = (255, 150, 0) # light blue


def main():
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
            #frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            #print("Frame size : ", frame_size)

            #ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))

            bev_frame = bev(frame)

            line_img = cv2.inRange(bev_frame, wh_low_bgr, wh_upp_bgr)

            line_contours, line_hier = cv2.findContours(line_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
            # 노이즈 제거를 위한 모폴로지 연산 적용
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            line_img = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, kernel, iterations=2)
  
            line_contours, line_hier = cv2.findContours(line_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 중심 선 그리기
            if line_contours:
                line_largest_contour = max(line_contours, key=cv2.contourArea)

                points_by_y = {}
                for point in line_largest_contour:
                    x, y = point[0]
                    if y in points_by_y:
                        points_by_y[y].append((x, y))
                    else:
                        points_by_y[y] = [(x, y)]

                midpoints = []
                for y, points in points_by_y.items():
                    if len(points) > 1:
                        x_values = [point[0] for point in points]
                        mid_x = int(np.mean(x_values))
                        midpoints.append((mid_x, y))


                # 중간 좌표 연결한 선 그리기
                for i in range(len(midpoints) - 1):
                    cv2.line(bev_frame, midpoints[i], midpoints[i + 1], (0, 255, 0), 2)

            cv2.drawContours(bev_frame, line_contours, -1, detected_line_color , 2 , cv2.LINE_8, line_hier)
    
            cv2.imshow("Camera", frame)
            cv2.imshow("Detection in Bird Eye View", bev_frame)

        if cv2.waitKey(1) & 0xFF == 27 :
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
