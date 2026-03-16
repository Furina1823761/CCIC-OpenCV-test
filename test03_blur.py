import cv2

image = cv2.imread("./img/plane.jpg")

# 高斯滤波
guass = cv2.GaussianBlur(image, (5,5), 0)
median = cv2.medianBlur(image, 5)

cv2.imshow("image",image)
cv2.imshow("guass", guass)
cv2.imshow("median", median)

cv2.waitKey()


