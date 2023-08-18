import cv2
import numpy as np
import matplotlib.pyplot as plt


# yellow_bgr = [60,255,255]
ye_low_bgr = np.array([40, 170, 170])
ye_upp_bgr = np.array([90, 230, 230])

detected_huddle_color = (255 , 100 , 255) # light violet

cap = cv2.VideoCapture(2)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Frame size : ", frame_size)


def bev(image) :

    # height, width = image.shape[:2]

    # ource = np.float32([[width/3, 0], [0, height], [width*2/3, 0], [width, height]])
    # destination = np.float32([[0,0], [0,height],[width,0] ,[width, height] ])

    # transform_matrix = cv2.getPerspectiveTransform(source, destination)
    # transformed_img = cv2.warpPerspective(image, transform_matrix, (width,height))

    transformed_img = image  # 버드아이뷰 안쓸 때

    return transformed_img



while True : 
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    bev_frame = bev(frame)
    
    huddle_img = cv2.inRange(bev_frame, ye_low_bgr, ye_upp_bgr)

    huddle_contours, huddle_hier = cv2.findContours(huddle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 노이즈 제거를 위한 모폴로지 연산 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    huddle_img = cv2.morphologyEx(huddle_img, cv2.MORPH_OPEN, kernel, iterations=2)

    huddle_contours, huddle_hier = cv2.findContours(huddle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(bev_frame, huddle_contours, -1, detected_huddle_color , 2 ,cv2.LINE_8, huddle_hier)
    cv2.imshow("Detection in Bird Eye View", bev_frame)

    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows


