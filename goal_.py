import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

rospy.init_node('goal_detection')

cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

cap = cv2.VideoCapture(3)

while not rospy.is_shutdown():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    # bgr
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([80, 80, 255])   
    red_mask = cv2.inRange(frame, lower_red, upper_red)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2, cv2.LINE_8, _)

    largest_contour = None
    largest_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > largest_area:
            largest_area = area
            largest_contour = contour
            
    if largest_contour is not None:

        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        
        cx = x + w // 2
        cy = y + h // 2
        
        print("Goal center (x, y):", cx, cy)

        error_x = cx - frame.shape[1] / 2  

        Kp = 0.001
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0
        cmd_vel_msg.angular.z = -Kp * error_x 
        cmd_vel_pub.publish(cmd_vel_msg)
        
        # 중심과 화면 중심을 연결하는 선 그리기
        cv2.line(frame, (cx, cy), (frame.shape[1] // 2, frame.shape[0] // 2), (255, 255, 255), 2)
        
        # (cx, cy)와 화면 중심에 점 그리기
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 5, (0, 255, 0), -1)
     
    cv2.imshow("Goal Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
