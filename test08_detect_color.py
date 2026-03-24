import cv2
import numpy as np
import torch

yellow =    [81, 174, 189] #BGR
red =       [47, 50, 164] #BGR
blue =      [181, 128, 33] #BGR
black =     [0, 0, 0] #BGR
white =     [255, 255, 255] #BGR

def rgb_to_hsv(color):
    # 归一化到 [0, 1]
    b, g, r = color[0] / 255.0, color[1] / 255.0, color[2]  / 255.0
    
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    
    # H
    if delta == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / delta % 6))
    elif max_c == g:
        h = (60 * ((b - r) / delta + 2))
    else:  # max_c == b
        h = (60 * ((r - g) / delta + 4))
    
    if h < 0:
        h += 360
    
    # S
    if max_c == 0:
        s = 0
    else:
        s = delta / max_c
    
    # V
    v = max_c
    hsv = np.stack([h, s * 255, v * 255], axis=-1)
    return hsv

def bgr2hsv(img):
    img_tensor = torch.from_numpy(img).float() / 255.0
    b = img_tensor[..., 0]
    g = img_tensor[..., 1]
    r = img_tensor[..., 2]

    max_val, _ = torch.max(img_tensor, dim=2)
    min_val, _ = torch.min(img_tensor, dim=2)
    delta = max_val - min_val
    # 计算色调
    h = torch.zeros_like(max_val)

    mask_r = (max_val == r) & (delta != 0)
    h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / delta[mask_r] % 6)

    mask_g = (max_val == g) & (delta != 0)
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)

    mask_b = (max_val == b) & (delta != 0)
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

    h = h % 360
    h = (h / 2).to(torch.uint8)  

    # 饱和度 S
    s = torch.zeros_like(max_val)
    mask_nonzero = (max_val != 0)
    s[mask_nonzero] = delta[mask_nonzero] / max_val[mask_nonzero]
    s = (s * 255).to(torch.uint8)

    v = (max_val * 255).to(torch.uint8)

    hsv = torch.stack([h, s, v], dim=-1)
    return hsv.numpy()


def get_balck_or_red_limits(color):

    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)


    if (color == red):
        lowerLimit1 = hsvC[0][0][0]+ 180 - 5 , 100, 100
        upperLimit1 =180, 255, 255
        lowerLimit2 =0, 100, 100
        upperLimit2= hsvC[0][0][0] + 5, 255, 255


        lowerLimit1 = np.array(lowerLimit1, dtype = np.uint8)
        upperLimit1 = np.array(upperLimit1, dtype = np.uint8)
        lowerLimit2 = np.array(lowerLimit2, dtype = np.uint8)
        upperLimit2= np.array(upperLimit2, dtype = np.uint8)
    else:
        lowerLimit1 = 0, 0, 0
        upperLimit1 = 179, 255, 20
        lowerLimit2 = 0, 0, 0
        upperLimit2= 179, 60, 60   

        lowerLimit1 = np.array(lowerLimit1, dtype = np.uint8)
        upperLimit1 = np.array(upperLimit1, dtype = np.uint8)
        lowerLimit2 = np.array(lowerLimit2, dtype = np.uint8)
        upperLimit2= np.array(upperLimit2, dtype = np.uint8)


    return (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2)

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    if (color != black and color != white):
        lowLimit = hsvC[0][0][0] - 5, 100, 100
        highLimit = hsvC[0][0][0] + 5, 255, 255

        lowLimit = np.array(lowLimit, dtype = np.uint8)
        highLimit = np.array(highLimit, dtype = np.uint8)   
    elif (color == white):
        lowLimit = 0, 0, 100
        highLimit = 179, 23, 255

        lowLimit = np.array(lowLimit, dtype = np.uint8)
        highLimit = np.array(highLimit, dtype = np.uint8)



    print(hsvC[0][0])
    print(rgb_to_hsv(color))

    return lowLimit, highLimit

# 识别颜色的物体
def main():

    cap = cv2.VideoCapture(0)
    color = red
    if (color == red or color == black):
        (lowerLimit1, upperLimit1), (lowerLimit2, upperLimit2) = get_balck_or_red_limits(color)
    else:
        lowLimit, highLimit = get_limits(color)

    while True:
        ret, frame = cap.read()

        hsvImg = bgr2hsv(frame)
        # hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if (color == red or color == black):
            mask1 = cv2.inRange(hsvImg, lowerLimit1, upperLimit1)
            mask2 = cv2.inRange(hsvImg, lowerLimit2, upperLimit2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
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