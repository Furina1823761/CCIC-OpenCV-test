import cv2
import os

VIDEO_ROOT = "video"
SAVE_ROOT = "dataset/train"
FRAME_INTERVAL = 2
IMG_SIZE = 128

def process_video(video_path, save_dir, start_count):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开 {video_path}")
        return start_count

    frame_id = 0
    count = start_count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_INTERVAL == 0:
            h, w, _ = frame.shape
            min_dim = min(h, w)

            # 中心裁剪
            start_x = w // 2 - min_dim // 2
            start_y = h // 2 - min_dim // 2
            crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # resize + 灰度
            img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            filename = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(filename, gray)

            count += 1

        frame_id += 1

    cap.release()
    print(f"✅ 处理完成: {video_path}")
    return count


# ===== 主流程 =====
for cls in os.listdir(VIDEO_ROOT):
    cls_path = os.path.join(VIDEO_ROOT, cls)

    if not os.path.isdir(cls_path):
        continue

    save_dir = os.path.join(SAVE_ROOT, cls)
    os.makedirs(save_dir, exist_ok=True)

    count = 0

    for file in os.listdir(cls_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(cls_path, file)
            count = process_video(video_path, save_dir, count)

    print(f"🎯 类别 {cls} 共生成 {count} 张图片")