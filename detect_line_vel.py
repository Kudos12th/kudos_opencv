import cv2
import numpy as np
import math
from geometry_msgs.msg import Twist
import rospy


last_angular_vel = 0.0
last_linear_vel = 0.0

cap = cv2.VideoCapture(2)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Frame size : ", frame_size)

def calculate_velocities(centers, image_width) :
    global last_angular_vel, last_linear_vel
    if len(centers) >= 2 :
        center_x = image_width // 2
        line_center_x = sum(point[0] for point in centers) // len(centers)
        angle_error = math.atan2(center_x - line_center_x, image_width) * 2.0

        max_angular_vel = 0.15
        max_linear_vel = 0.05

        angular_vel = max_angular_vel * angle_error
        linear_vel = max_linear_vel * (1.0 - abs(angle_error))

        last_angular_vel = angular_vel
        last_linear_vel = linear_vel

        return angular_vel, linear_vel
    
    else:
        return last_angular_vel,last_linear_vel


def find_white(image) :
    wh_low_bgr = np.array([100, 100 , 100])
    wh_upp_bgr = np.array([255, 255, 255])
    line_img = cv2.inRange(image, wh_low_bgr, wh_upp_bgr)

    return line_img


# bird eye view
def bev(image) :

    # height, width = image.shape[:2]
    width = 640 #320
    height = 480 #240
    
    # source = np.float32([[280, 440], [270, 480], [360, 440], [370, 480]])
    source = np.float32([[200, 430], [190, 480], [440, 430], [450, 480]])
    destination = np.float32([[0,0], [0,height],[width,0] ,[width, height] ])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    transformed_img = cv2.warpPerspective(image, transform_matrix, (640,480))

    # transformed_img = image  # 버드아이뷰 안쓸 때

    return transformed_img

def noise(image) :
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    noise_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

    return noise_img


def window(image,out_img) :

    margin = 50
    nwindows = 5
    window_height = int(image.shape[0]/nwindows)
    line_color = [255,0,0]
    window_color = [0, 0, 255]
    center_color = [0, 255, 0]
    thickness = 2
    centers = []

    line_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, line_contours, -1, line_color, thickness, cv2.LINE_8, _)

    for w in range(nwindows) :
        win_y_low = image.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = image.shape[0] - w * window_height # winodw 아랫부분

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

    cv2.imshow('detection in Bird Eye View', out_img)

    return centers



while True :
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    print("Press 'esc' to quit.")

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    
    bev_img = bev(frame)
    noise_img = noise(bev_img)
    wh_img = find_white(noise_img)

    centers = window(wh_img, bev_img)

    angular_vel, linear_vel = calculate_velocities(centers, bev_img.shape[1])

    twist = Twist()
    twist.angular.z = angular_vel
    twist.linear.x = linear_vel
    cmd_vel_pub.publish(twist)

    cv2.imshow('camera', frame)

    print("Press 'esc' to quit.")
    if cv2.waitKey(1) & 0xFF == 27 :
        break


cap.release()
cv2.destroyAllWindows()

