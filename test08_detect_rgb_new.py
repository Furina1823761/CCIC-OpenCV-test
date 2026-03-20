import cv2
import numpy as np

# ---------- 颜色名称与显示颜色 ----------
COLOR_NAMES = [
    "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"
]
# 显示圆点的颜色（BGR）
DOT_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (255, 0, 255)   # Magenta
]

# ---------- 系数搜索函数（与原一致） ----------
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
    param 包含：
        frame: 当前帧
        samples: 样本列表，长度为6，每个元素是列表
        current_idx: 当前要采集的颜色索引
        display: 用于显示的图像副本
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = param['frame'][y, x]
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        idx = param['current_idx']
        param['samples'][idx].append(rgb)
        # 在显示图上画对应颜色的圆点
        cv2.circle(param['display'], (x, y), 5, DOT_COLORS[idx], -1)
        print(f"{COLOR_NAMES[idx]} 样本 {len(param['samples'][idx])}: RGB{rgb}")

# ---------- 主程序 ----------
def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera")

    # 初始化六种颜色的样本列表
    samples = [[] for _ in range(6)]
    current_color = 0  # 当前采集的颜色索引 0~5

    param = {
        'frame': None,
        'samples': samples,
        'current_idx': current_color,
        'display': None
    }
    cv2.setMouseCallback("Camera", mouse_callback, param)

    print("操作说明：")
    print("  - 数字键 1~6 选择要采集的颜色（1:红,2:绿,3:蓝,4:黄,5:白,6:黑）")
    print("  - 鼠标左键点击对应颜色的区域，添加样本")
    print("  - 按空格键计算所有颜色的最佳分类器")
    print("  - 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        param['frame'] = frame
        param['display'] = display
        param['current_idx'] = current_color

        # 显示当前选中的颜色和已采集样本数
        for i, name in enumerate(COLOR_NAMES):
            text = f"{name}: {len(samples[i])}"
            color = (0, 255, 0) if i == current_color else (255, 255, 255)
            cv2.putText(display, text, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # 检查每种颜色是否有足够样本
            min_samples = 2  # 每种颜色至少需要2个样本
            insufficient = [i for i, lst in enumerate(samples) if len(lst) < min_samples]
            if insufficient:
                print(f"以下颜色样本不足 {min_samples} 个: {[COLOR_NAMES[i] for i in insufficient]}")
                continue

            print("\n========== 训练所有颜色分类器 ==========")
            best_overall_score = -1
            best_color_info = None
            for target_idx in range(6):
                pos = samples[target_idx]
                neg = []
                for i in range(6):
                    if i != target_idx:
                        neg.extend(samples[i])
                
                abc, score = find_best_linear_combination(pos, neg, step=4)
                a, b, c = abc
                # 计算正负样本的 v 值
                pos_vals = [a*r + b*g + c*b_ for (r,g,b_) in pos]
                neg_vals = [a*r + b*g + c*b_ for (r,g,b_) in neg]
                mean_pos = np.mean(pos_vals)
                mean_neg = np.mean(pos_vals)  # 这里应该是 neg_vals，修正
                mean_neg = np.mean(neg_vals)
                threshold = (mean_pos + mean_neg) / 2

                print(f"\n--- {COLOR_NAMES[target_idx]} 作为目标 ---")
                print(f"  系数: a={a}, b={b}, c={c}")
                print(f"  Fisher分数: {score:.2f}")
                print(f"  正样本 v 均值: {mean_pos:.2f}")
                print(f"  负样本 v 均值: {mean_neg:.2f}")
                print(f"  推荐阈值: {threshold:.2f}")
                if mean_pos > mean_neg:
                    print(f"  判定规则: v > {threshold:.2f} 为目标颜色")
                else:
                    print(f"  判定规则: v < {threshold:.2f} 为目标颜色")

                if score > best_overall_score:
                    best_overall_score = score
                    best_color_info = (COLOR_NAMES[target_idx], abc, threshold, mean_pos > mean_neg)

            print("\n========== 最佳分类器 ==========")
            if best_color_info:
                name, (a,b,c), th, pos_greater = best_color_info
                print(f"目标颜色: {name}")
                print(f"系数: a={a}, b={b}, c={c}")
                print(f"阈值: {th:.2f}")
                if pos_greater:
                    print(f"判定规则: v = {a}*R + {b}*G + {c}*B  > {th:.2f} 则为 {name}")
                else:
                    print(f"判定规则: v = {a}*R + {b}*G + {c}*B  < {th:.2f} 则为 {name}")
                print("你可以将该规则用于实时颜色识别。")
            print("===================================\n")

        elif ord('1') <= key <= ord('6'):
            current_color = key - ord('1')
            print(f"当前采集颜色切换到: {COLOR_NAMES[current_color]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()