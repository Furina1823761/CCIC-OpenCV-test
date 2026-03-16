import cv2
import numpy as np

image = np.zeros([512,512,3], dtype=np.uint8)



cv2.line(image, (110,200), (300,300), (255,0,0), 2)
cv2.imshow("image", image)
cv2.waitKey()