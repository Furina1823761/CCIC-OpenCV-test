import cv2 

image =  cv2.imread("./img/green_block.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(gray, cv2.CV_64F)

canny = cv2.Canny(gray, 50, 200)


cv2.imshow("gray", gray)
cv2.imshow("laplacian", laplacian)

cv2.imshow("canny", canny)


cv2.waitKey()
