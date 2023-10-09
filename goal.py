import cv2
import numpy as np

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    moments = cv2.moments(red_mask)
    
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        print("goal center (x, y):", cx, cy)
        
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    
    result_frame = cv2.bitwise_and(frame, frame, mask=red_mask)
    
    cv2.imshow("Goal Detection", frame)
    cv2.imshow("RED", red_mask)
    
    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()
