import cv2

print(cv2.getVersionString())

image = cv2.imread("./img/fufu01.jpeg")
print(image.shape)

cv2.imshow("image", image)
cv2.waitKey()