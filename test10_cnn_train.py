import torch
import torch.nn as nn 
import cv2
import numpy as np

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
    

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

full_dataset = ImageFolder("dataset/train", transform=transform)

# 80% 训练 + 20% 验证
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNNet(num_classes=3).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    class_weights = torch.tensor([0.9, 1.0, 1.0]) 
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(20):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ✅ 每轮评估
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "cnn_weights.pth")
    print("Model saved!")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            pred = torch.argmax(output, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

if __name__ == "__main__":
    train()