# 相机捕获,做灰度处理和边缘检测
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("camera", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    canny = cv2.Canny(median, 100, 200)
    cv2.imshow("canny", canny)

    key = cv2.waitKey(1) 
    if key != -1:
        break
