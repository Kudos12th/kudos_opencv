import cv2
import numpy as np
import math
from geometry_msgs.msg import Twist
import rospy

from std_msgs.msg import Int32

last_angular_vel = 0.0
last_linear_vel = 0.0
zero_cnt = 0
h_stop = 0 
slope_cnt = 0

rospy.init_node('video_subscriber_node', anonymous=True)
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
h_stop_pub = rospy.Publisher('/h_stop', Int32, queue_size=1)

cap = cv2.VideoCapture(2)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Frame size : ", frame_size)


def bev(image):
    width=640
    height=480
    # view=[[170, 330], [130, 480], [470, 330], [510, 480]] #좌상 좌하 우상 우하

    #280
    view=[[130, 330], [90, 480], [430, 330], [470, 480]] #좌상 좌하 우상 우하


    source = np.float32(view)
    destination = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (width, height))

    return transformed_img



def find_yellow(img) :
    # bgr
    ye_low_bgr = np.array([30, 170, 90])
    ye_upp_bgr = np.array([140, 255, 180])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel, iterations=2)
    yellow_img = cv2.inRange(img, ye_low_bgr, ye_upp_bgr)
    
    huddle_contours, huddle_hier = cv2.findContours(yellow_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return huddle_contours, huddle_hier



def huddle_detect(huddle_contours, huddle_hier, out_img):
    huddle_centers = []

    for contour in huddle_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0 :
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            huddle_centers.append((cX, cY))

    sorted_centers = sorted(huddle_centers, key=lambda x: x[1], reverse=True)
    for idx, center in enumerate(sorted_centers, start = 1):
        cX, cY = center
        #print(f"Huddle {idx}: X = {cX}, Y = {cY}")
    if sorted_centers:
        cv2.circle(out_img, sorted_centers[0], 10, (0, 255, 0), -1)

    cv2.drawContours(out_img, huddle_contours, -1, (255, 100, 255), 2, cv2.LINE_8, huddle_hier)
    cv2.imshow("Camera with Huddle", out_img)

    return sorted_centers


def cal_slope(contours,img) :
    largest_contour = max(contours, key=cv2.contourArea)
    for point in largest_contour:
        
        left_points = []  
        right_points = [] 
        min_x = float('inf') 
        max_x = float('-inf')
        left_yellow = None
        right_yellow = None

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
        left_yellow = max(left_points, key=lambda point: point[1])  # 왼쪽 점 중 아래

    if right_points:
        right_yellow = max(right_points, key=lambda point: point[1])  # 오른쪽 점 중 아래
        
    if left_yellow :
        if right_yellow :
            
            cv2.line(img, left_yellow, right_yellow, (255,0,0), (2))

            delta_x = right_yellow[0] - left_yellow[0]
            if delta_x != 0:
                delta_y = right_yellow[1] - left_yellow[1]
                slope = delta_y / delta_x
                print('slope :', slope)
            else:
                slope = None

            return slope




def find_white(img):
    wh_low_bgr = np.array([230, 230, 230])
    wh_upp_bgr = np.array([255, 255, 255])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel, iterations=2)
    white_img = cv2.inRange(img, wh_low_bgr, wh_upp_bgr)

    line_contours, line_hier = cv2.findContours(white_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return line_contours, line_hier



def largest_center(line_contours, line_hier, out_img):
    dst_point = None
    cv2.drawContours(out_img, line_contours, -1, [ 0, 0, 255], 2, cv2.LINE_8, line_hier)
    if line_contours:
        largest_line_contour = max(line_contours, key=cv2.contourArea)
        dst_point = tuple(largest_line_contour[0][0])
        cv2.circle(out_img, dst_point, 5, (0, 255, 0), -1)
    cv2.imshow("Detection in Bird Eye View(largest_center)", out_img)
    return dst_point



def top_center(line_contours, line_hier, out_img) :
    dst_point = None
    cv2.drawContours(out_img, line_contours, -1, [ 0, 0, 255], 2, cv2.LINE_8, line_hier)
    
    if line_contours:
        dst_point = tuple(line_contours[0][0][0])
        cv2.circle(out_img, dst_point, 5, (0, 255, 0), -1)
    cv2.imshow("Detection in Bird Eye View(top_center)", out_img)

    return dst_point


def window(line_contours, line_hier,out_img) :
    margin = 50
    nwindows = 12
    window_height = int(out_img.shape[0]/nwindows)
    line_color = [255,0,0]
    window_color = [0, 0, 255]
    center_color = [0, 255, 0]
    thickness = 2
    centers = []
    dst_point = None

    cv2.drawContours(out_img, line_contours, -1, line_color, thickness, cv2.LINE_8, line_hier)

    for w in range(nwindows) :
        win_y_low = out_img.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = out_img.shape[0] - w * window_height # winodw 아랫부분

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
                centers.append((cX, cY)) 

                win_x_low = cX - margin
                win_x_high = cX + margin

                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), window_color, thickness)

    if len(centers) > 1:
        for i in range(len(centers) -1):
            cv2.line(out_img, centers[i], centers[i+1], center_color, thickness)

    # centers[-1]과 centers[0]의 방향을 생각해서 각각 좌회전/우회전만 저장할 수 있는지!
    if centers:
        dst_point = centers[-1]
        cv2.circle(out_img, dst_point, 5, (0, 255, 0), -1)

    cv2.imshow('Detection in Bird Eye View(window)', out_img)

    return dst_point



def cal_vel(img_width, dst_point):
    global last_angular_vel, last_linear_vel

    center_x = img_width // 2

    if dst_point is not None:
        dst_x = dst_point[0]
        angle_error = math.atan2(center_x - dst_x, img_width) * 2.0

        max_angular_vel = 0.3
        max_linear_vel = 0.05

        angular_vel = max_angular_vel * angle_error
        linear_vel = max_linear_vel * (1.0 - abs(angle_error))

        last_angular_vel = angular_vel
        last_linear_vel = linear_vel
        
        # print("destination error", center_x - dst_x)
        # print("dst_x =", dst_x)
        return angular_vel, linear_vel
    else:
        # print("---No line---")
        # return 0.0, 0.0
        return last_angular_vel, last_linear_vel
    


while True :

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))

    bev_img = bev(frame)
    line_contours, line_hier = find_white(bev_img)

    dst_point = window(line_contours, line_hier, bev_img)
    # dst_point = top_center(line_contours, line_hier, bev_img)
    # dst_point = largest_center(line_contours, line_hier, bev_img)

    huddle_contours, huddle_hier = find_yellow(frame)
    huddle_centers = huddle_detect(huddle_contours, huddle_hier, frame)

    if huddle_centers:
        if huddle_centers[0][1] >= 460:

            slope = cal_slope(huddle_contours, frame)
            # print("hurdle stop")

            angular_vel, linear_vel = 0.0, 0.0

            zero_cnt += 1

            if zero_cnt > 9:

                angular_vel, linear_vel = slope, 0.0

                if -0.1 < slope < 0.1 :
                    print('slope under 0.1')
                    slope_cnt += 1

                    if slope_cnt == 9 :
                        h_stop = 1
                        print('**Hurdle STOP**')

                    elif slope_cnt == 10 :
                        slope_cnt = 0
                        zero_cnt = 0  
                        h_stop = 0

        else:
            angular_vel, linear_vel = cal_vel(bev_img.shape[1], dst_point)
    else:
        angular_vel, linear_vel = cal_vel(bev_img.shape[1], dst_point)

    twist = Twist()
    twist.angular.z = angular_vel
    twist.linear.x = linear_vel
    cmd_vel_pub.publish(twist)

    msg = Int32()
    msg.data = h_stop
    h_stop_pub.publish(msg)

    if cv2.waitKey(1) & 0xFF == 27 :
            break
    
cap.release()
cv2.destroyAllWindows()