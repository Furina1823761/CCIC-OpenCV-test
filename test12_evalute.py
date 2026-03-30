import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import time

class CNNNet(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        # 输入: [batch, 1, 256, 256]
        self.conv1 = torch.nn.Conv2d(
            in_channels= 1, # 灰度图，1维
            out_channels= 16, # 输出通道数
            kernel_size= 3,
            padding= 1
        )# 输出: [batch, 16, 256, 256]
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(2, 2) # 输出: [batch, 16, 128, 128]
        

        self.conv2 = torch.nn.Conv2d(
            in_channels = 16,
            out_channels= 32,
            kernel_size= 3,
            padding= 1,
        )# 输出: [batch, 32, 128, 128]
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2,2) # 输出: [batch, 32, 64, 64]

        self.conv3 = torch.nn.Conv2d(
            in_channels = 32,
            out_channels= 64,
            kernel_size= 3,
            padding= 1,
        )# 输出: [batch, 64, 64, 64]
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool3 = torch.nn.MaxPool2d(2,2) # 输出: [batch, 64, 32, 32]

        self.conv4 = torch.nn.Conv2d(
            in_channels = 64,
            out_channels= 128,
            kernel_size= 3,
            padding= 1,
        )# 输出: [batch, 128, 32, 32]
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(2,2) # 输出: [batch, 128, 16, 16]
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()


        self.fc1 = torch.nn.Linear(128, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)

        x = self.global_pool(x)
        x = self.flatten(x)   

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
 

# 类别名（对应训练时的顺序）
class_names = ["cone", "cube", "cylinder","others"]  

def real_time_detect(model_path="cnn_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNet(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    cap = cv2.VideoCapture("video/test01.mp4")  # 摄像头
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    

    IMG_SIZE = 256

    while True:
        ret, frame = cap.read()
        time.sleep(1/30)

        # 裁剪中心正方形
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = w // 2 - min_dim // 2
        start_y = h // 2 - min_dim // 2
        crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]



        # 灰度 + resize
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        cv2.imshow("img",img)

        # 转成 tensor [1,1,256,256]
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
        x = x.to(device)

        # 推理
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
            label = class_names[pred]
        cv2.putText(frame, f"Prediction: {output}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # 显示
        cv2.putText(frame, f"Prediction: {label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("CNN Real-time Detection", frame)

        # 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def evalute(model, val_loader, device, class_names):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 数据预处理（与训练完全一致）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 3. 加载整个训练数据集
    full_dataset = ImageFolder("dataset/train", transform=transform)
    class_names = full_dataset.classes   # 自动获取文件夹顺序，如 ['cone','cube','cylinder','others']
    print("类别顺序:", class_names)

    # 4. 划分训练集和验证集（例如 80% 训练，20% 验证）
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"验证集大小: {val_size}")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # 5. 加载训练好的模型权重
    model = CNNNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("cnn_weights.pth", map_location=device))
    print("模型权重加载成功")

    # 6. 评估
    evalute(model, val_loader, device, class_names)
    

if __name__ == "__main__":
    # 训练完成后调用实时检测
    real_time_detect("cnn_weights.pth")
    # main()