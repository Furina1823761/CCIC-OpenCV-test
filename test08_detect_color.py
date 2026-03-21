import cv2
import numpy as np



def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowLimit = hsvC[0][0][0] - 5, 150, 150
    highLimit = hsvC[0][0][0] + 5, 255, 255

    lowLimit = np.array(lowLimit, dtype = np.uint8)
    highLimit = np.array(highLimit, dtype = np.uint8)

    return lowLimit, highLimit


def main():
    yellow = [0, 255, 255] #BGR
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lowLimit, highLimit = get_limits(yellow)
        mask = cv2.inRange(hsvImg, lowLimit, highLimit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:   
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
