import cv2

cap = cv2.VideoCapture(0)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Frame size : ", frame_size)

ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = frame[y, x]
        print(f'BGR Color at ({x}, {y}): {bgr_color}')

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    # 이미지를 화면에 표시합니다.
    cv2.imshow('Image', frame)
    
    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
