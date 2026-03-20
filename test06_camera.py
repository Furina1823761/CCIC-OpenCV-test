# 相机捕获,做灰度处理和边缘检测
import cv2
import numpy as np

# ---------- 新增：颜色判断函数 ----------
def get_color_name(bgr):
    """根据BGR平均值返回颜色名称（简单阈值）"""
    b, g, r = bgr
    color_list = [0, 0, 0, 0, 0, 0]
    v_green = -120 * r + 52 * g + 44 * b
    if(v_green > -4642.76):
        color_list[0] = 1
    else:
        color_list[0] = 0
    v_red = -68 * r + 120 * g - 44 * b
    if(v_red < -1421.20):
        color_list[1] = 1
    else:
        color_list[1] = 0
    v_yellow = -4 * r + 128 * g - 124 * b 
    if(v_yellow > 6931.33):
        color_list[2] = 1
    else:
        color_list[2] = 0
    v_blue = -64 * r -72 * g + 124 * b 
    if(v_blue > 2379.49):
        color_list[3] = 1
    else:
        color_list[3] = 0
    v_white = -32 * r + 20 * g - 80 * b 
    if(v_white < -14154.73):
        color_list[4] = 1
    else:
        color_list[4] = 0
    v_black = -36 * r - 80 * g + 20 * b
    if(v_black > -7089.97):
        color_list[5] = 1
    else:
        color_list[5] = 0

    # return "green:{} red:{} yellow:{}\nblue:{} white:{} black:{}".format(color_list[0], color_list[1], color_list[2], color_list[3], color_list[4], color_list[5])
    # if(color_list[0] == 1):
    #     return "Green"
    if(color_list[1] == 1):
        return "Red"
    elif(color_list[2] == 1):
        return "Yellow"
    elif(color_list[3] == 1):
        return "Blue"
    elif(color_list[4] == 1):
        return "White"
    elif(color_list[5] == 1):
        return "Black"
    elif(color_list[0] == 1):
        return "Green"
    else:
        return "Unknown"
    
    # min_rgb = min(r, g, b)
    # max_rgb = max(r, g, b)

    


    # if min_rgb > 200:
    #     return "White"
    # elif max_rgb < 50:
    #     return "Black"
    # else:
    #     b -= min_rgb
    #     r -= min_rgb
    #     g -= min_rgb

    #     if(b == 0):
    #         p = r / g 
    #         if(p > 1.5):
    #             return "Red"
    #         elif(p < 0.67):
    #             return "Green"
    #         else:
    #             return "Yellow"
            
    #     elif(g == 0):
    #         p = r / b   
    #         if(p > 1.5):
    #             return "Red"
    #         elif(p < 0.67):
    #             return "Blue"
    #         else:
    #             return "Purple"
    #     else:
    #         p = g / b   
    #         if(p > 1.5):
    #             return "Green"
    #         elif(p < 0.67):
    #             return "Blue"
    #         else:
    #             return "Lightblue"

    # if r > 150 and g < 100 and b < 100:
    #     return "Red"
    # elif g > 150 and r < 100 and b < 100:
    #     return "Green"
    # elif b > 150 and r < 100 and g < 100:
    #     return "Blue"
    # elif r > 150 and g > 150 and b < 100:
    #     return "Yellow"
    # elif r > 150 and g < 150 and b > 150:
    #     return "Magenta"
    # elif r < 100 and g > 100 and b > 100:
    #     return "Cyan"
    # elif r > 200 and g > 200 and b > 200:
    #     return "White"
    # elif r < 50 and g < 50 and b < 50:
    #     return "Black"
    # else:
    #     return "Unknown"

# ---------- 新增：形状颜色识别函数 ----------
def analyze_contour(contour, frame):
    """输入轮廓和原始彩色图，返回形状名称、颜色名称和绘制用的边框"""
    area = cv2.contourArea(contour)
    if area < 500:  # 忽略小噪点
        return None, None, None

    # 多边形逼近
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)

    # 圆度
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    # 最小外接矩形
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width > 0 and height > 0:
        aspect_ratio = max(width, height) / min(width, height)
        rect_area = width * height
        rect_ratio = area / rect_area if rect_area > 0 else 0
    else:
        aspect_ratio = 0
        rect_ratio = 0

    # 形状分类
    shape = "Unknown"
    if vertices == 3:
        shape = "Cone (triangle)"
    elif vertices == 4:
        if aspect_ratio < 1.2 and rect_ratio > 0.8:
            shape = "Cube (square)"
        else:
            shape = "Cylinder (rectangle)"
    elif circularity > 0.7:
        shape = "Cylinder/Cone (circle)"
    else:
        shape = "Unknown"

    # 提取轮廓区域的平均颜色
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)  # 填充轮廓
    mean_bgr = cv2.mean(frame, mask=mask)[:3]
    color_name = get_color_name(mean_bgr)

    # 获取外接矩形用于文字位置
    x, y, w, h = cv2.boundingRect(contour)
    return shape, color_name, (x, y-10)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("camera", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    median = cv2.medianBlur(gray, 5)
    canny = cv2.Canny(median, 100, 100)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(canny , cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closed", closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        shape, color_name, text_pos = analyze_contour(cnt, frame)
        if shape is None:  # 跳过小轮廓
            continue
        # 绘制轮廓
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        # 标注形状和颜色
        if text_pos:
            cv2.putText(frame, f"{shape} {color_name}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)




    # corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 2)
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(frame, (int(x),int(y)), 3, (255,0,0), -1)
    cv2.imshow("camera", frame)

    key = cv2.waitKey(1) 
    if key != -1:
        break
    
