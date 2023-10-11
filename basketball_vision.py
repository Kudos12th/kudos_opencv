import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist

from std_msgs.msg import Int32


rospy.init_node('ball_walk')
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
b_stop_pub = rospy.Publisher('/b_stop', Int32, queue_size=1)

flag = 1
blue_cnt = 0
slope_cnt = 0
red_cnt = 0
b_stop = 0



cap = cv2.VideoCapture(0)

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


def cal_slope(contours,img) :
    largest_contour = max(contours, key=cv2.contourArea)
    for point in largest_contour:
        
        left_points = []  
        right_points = [] 
        min_x = float('inf') 
        max_x = float('-inf')
        left_blue = None
        right_blue = None

        for point in largest_contour:
            x, y = point[0] 
            if x < min_x:
                min_x = x 
                left_points = [(x, y)]  
            elif x == min_x:
                left_points.append((x, y))

        for point in largest_contour:
            x, y = point[0] 
            if x > max_x:
                max_x = x 
                right_points = [(x, y)]  
            elif x == max_x:
                right_points.append((x, y))

    if left_points:
        left_blue = min(left_points, key=lambda point: point[1])  
    if right_points:
        right_blue = min(right_points, key=lambda point: point[1]) 
        
    if left_blue :
        if right_blue :

            cv2.line(img, left_blue, right_blue, (0,0,255), (2))

            delta_x = right_blue[0] - left_blue[0]
            if delta_x != 0:
                delta_y = right_blue[1] - left_blue[1]
                slope = delta_y / delta_x
                print('slope :', slope)
            else:
                slope = None

            return slope


def find_red(img) :

    # bgr
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([80, 80, 255])   
    red_mask = cv2.inRange(img, lower_red, upper_red)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if red_contours :
        cv2.drawContours(img, red_contours, -1, (0, 0, 255), 2, cv2.LINE_8, _)
        largest_red = max(red_contours, key=cv2.contourArea)

        if largest_red is not None:

            x, y, w, h = cv2.boundingRect(largest_red)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            
            cx = x + w // 2
            cy = y + h // 2
            
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            print("Goal center (x, y):", cx, cy)

        return cx
    else :
        return None


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))


    # 파란 바닥이 보이면 멈추기
    if flag == 1 :
        frame_blue_y = 240
        cv2.line(frame, (frame_blue_y,0), (frame_blue_y,480), (255,255,255), (2))

        blue_contours, blue_hier = find_blue(frame)
        cv2.drawContours(frame, blue_contours, -1, (0, 255, 0), 2, cv2.LINE_8, blue_hier)

        if blue_contours:
            blue_y = [point[0][1] for contour in blue_contours for point in contour]
            max_blue_y = max(blue_y)
            slope = cal_slope(blue_contours, frame)
            cv2.line(frame, (0,max_blue_y), (640,max_blue_y), (255,0,0), (2))
            
            if max_blue_y >= frame_blue_y :
                angular_vel = 0.0
                linear_vel = 0.0
                blue_cnt += 1
                print('max_blue_y = ', max_blue_y,'\n STOP')
                if blue_cnt > 5:
                    if -3.0 < slope < 3.0:
                        angular_vel = slope * 0.1
                        linear_vel =  0.0
                        if -1.0 < slope < 1.0:
                            angular_vel = slope * 0.3
                            linear_vel = 0.0
                            if -0.3 < slope < 0.3:
                                angular_vel= slope 
                                linear_vel = 0.0
                                if -0.1 < slope < 0.1 :
                                    angular_vel = 0.0
                                    linear_vel = 0.0
                                    print('slope under 0.1')
                                    slope_cnt += 1
                                    if slope_cnt == 5 :
                                        # 파란색 앞에서 멈추기
                                        b_stop = 1
                                        print('************************************************\n************************************************\n*******************75cm STOP********************\n************************************************\n************************************************')
                                    elif slope_cnt == 6:
                                        slope_cnt == 0
                                        blue_cnt == 0
                                        b_stop = 0
                                        flag = 2

            else:
                angular_vel = 0.0
                linear_vel = 0.4
                print('max_blue_y = ', max_blue_y)
        else:
            angular_vel = 0.0
            linear_vel = 0.4
            print('No Blue')

    # 골대찾기
    elif flag == 2 :
        frame_goal_x = 200
        cv2.line(frame, (frame_goal_x,0), (frame_goal_x,480), (255,255,255), (2))
        cx = find_red(frame)
        if cx :
            error_x = cx - frame_goal_x
            print('X_error = ', error_x)
            if -500 < cx < 500 :
                angular_vel = 0.0007 * error_x
                linear_vel = 0
                if -300 < cx < 300 :
                    angular_vel = 0.0012 * error_x
                    linear_vel = 0.0
                    if -100 < cx < 100 :
                        angular_vel = 0.0035 * error_x
                        linear_vel = 0.0
                        if -50 < cx < 50 :
                            angular_vel = 0.007 * error_x
                            linear_vel = 0.0
                            if -20 < error_x < 20 :
                                angular_vel = 0.0
                                linear_vel = 0.0
                                red_cnt += 1
                                if red_cnt == 5 :
                                    # 농구공 던지기
                                    print("************************************************\n************************************************\n*******************GOAL STOP********************\n************************************************\n************************************************")
                                    b_stop = 2
                                    
                                elif red_cnt == 6 :
                                    b_stop = 0
                                    flag = 3
        else :
            print('NO GOAL')
            angular_vel = 0.0
            linear_vel = -0.02

    elif flag == 3 :

        angular_vel = 0.0
        linear_vel = 0.0


    twist = Twist()
    twist.angular.z = angular_vel
    twist.linear.x = linear_vel
    cmd_vel_pub.publish(twist)

    msg = Int32()
    msg.data = b_stop
    b_stop_pub.publish(msg)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()