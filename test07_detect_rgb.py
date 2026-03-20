import cv2
import numpy as np

# ---------- 颜色系数搜索函数（沿用之前的） ----------
def find_best_linear_combination(pos_samples, neg_samples, step=4):
    """
    暴力搜索最佳线性组合系数 (a, b, c)
    :param pos_samples: 正样本列表，每个元素为 (R, G, B)
    :param neg_samples: 负样本列表，每个元素为 (R, G, B)
    :param step: 步长
    :return: (best_a, best_b, best_c), best_score
    """
    best_score = -np.inf
    best_abc = None
    pos_arr = np.array(pos_samples, dtype=np.int32)
    neg_arr = np.array(neg_samples, dtype=np.int32)
    
    for a in range(-128, 129, step):
        for b in range(-128, 129, step):
            for c in range(-128, 129, step):
                pos_vals = a * pos_arr[:, 0] + b * pos_arr[:, 1] + c * pos_arr[:, 2]
                neg_vals = a * neg_arr[:, 0] + b * neg_arr[:, 1] + c * neg_arr[:, 2]
                
                mean_pos = np.mean(pos_vals)
                mean_neg = np.mean(neg_vals)
                var_pos = np.var(pos_vals) if len(pos_vals) > 1 else 0
                var_neg = np.var(neg_vals) if len(neg_vals) > 1 else 0
                
                if var_pos + var_neg == 0:
                    score = 0
                else:
                    score = (mean_pos - mean_neg) ** 2 / (var_pos + var_neg)
                
                if score > best_score:
                    best_score = score
                    best_abc = (a, b, c)
    return best_abc, best_score

# ---------- 鼠标回调函数 ----------
def mouse_callback(event, x, y, flags, param):
    """
    param 是一个字典，包含：
        frame: 当前帧
        pos_samples: 正样本列表
        neg_samples: 负样本列表
        colors: 用于显示的图像副本
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键：正样本
        bgr = param['frame'][y, x]  # 注意OpenCV是BGR顺序
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # 转换为RGB
        param['pos_samples'].append(rgb)
        # 在显示图上画绿色圆点
        cv2.circle(param['display'], (x, y), 5, (0, 255, 0), -1)
        print(f"正样本 {len(param['pos_samples'])}: RGB{rgb}")

    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键：负样本
        bgr = param['frame'][y, x]
        rgb = (int (bgr[2]), int(bgr[1]), int(bgr[0]))
        param['neg_samples'].append(rgb)
        cv2.circle(param['display'], (x, y), 5, (0, 0, 255), -1)
        print(f"负样本 {len(param['neg_samples'])}: RGB{rgb}")

# ---------- 主程序 ----------
def main():
    cap = cv2.VideoCapture(0)


    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Camera")
    pos_samples = []
    neg_samples = []
    param = {
        'frame': None,
        'pos_samples': pos_samples,
        'neg_samples': neg_samples,
        'display': None
    }
    cv2.setMouseCallback("Camera", mouse_callback, param)

    print("操作说明：")
    print("  - 左键点击目标颜色区域 -> 添加正样本")
    print("  - 右键点击背景区域 -> 添加负样本")
    print("  - 按空格键计算最佳系数")
    print("  - 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 在frame副本上绘制样本点（避免影响原图采集）
        display = frame.copy()
        param['frame'] = frame      # 用于鼠标回调提取颜色
        param['display'] = display  # 用于绘制圆点

        # 显示当前样本数量
        cv2.putText(display, f"Pos: {len(pos_samples)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Neg: {len(neg_samples)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):  # 空格键计算
            if len(pos_samples) < 2 or len(neg_samples) < 2:
                print("正负样本各至少需要2个，请继续采集")
                continue

            print("正在计算最佳系数...")
            best_abc, score = find_best_linear_combination(pos_samples, neg_samples)
            a, b, c = best_abc

            # ---------- 新增：计算正负样本的v值及阈值 ----------
            pos_vals = [a*int(r) + b*int(g) + c*int(b_) for (r,g,b_) in pos_samples]  # 注意变量名冲突，用b_表示B
            neg_vals = [a*int(r) + b*int(g) + c*int(b_) for (r,g,b_) in neg_samples]

            mean_pos = np.mean(pos_vals)
            mean_neg = np.mean(neg_vals)
            std_pos = np.std(pos_vals)
            std_neg = np.std(neg_vals)

            # 简单阈值：两类均值的中心点
            threshold = (mean_pos + mean_neg) / 2

            # 输出完整信息
            print("\n========== 训练结果 ==========")
            print(f"最佳系数: a={a}, b={b}, c={c}")
            print(f"Fisher分数: {score:.2f}")
            print(f"正样本 v 均值: {mean_pos:.2f} ± {std_pos:.2f}")
            print(f"负样本 v 均值: {mean_neg:.2f} ± {std_neg:.2f}")
            print(f"推荐阈值: {threshold:.2f}")
            print("==============================\n")
            print("使用说明：")
            print("  v = a*R + b*G + c*B")
            if mean_pos > mean_neg:
                print(f"  若 v > {threshold:.2f} 则判定为目标颜色，否则为背景")
            else:
                print(f"  若 v < {threshold:.2f} 则判定为目标颜色，否则为背景")

            # 可选：将系数和阈值保存到文件
            # with open("color_params.txt", "w") as f:
            #     f.write(f"{a} {b} {c} {threshold} {1 if mean_pos>mean_neg else 0}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()