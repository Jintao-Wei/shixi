图像分类项目学习笔记
项目结构
核心模块解析
1. 模型架构 (alex.py)
```python class Alex(nn.Module): def init(self): super().init() self.model = nn.Sequential( # 卷积层1: 3→64通道，保持32x32尺寸 nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # 32x32 → 16x16

        # 卷积层2: 64→192通道
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 16x16 → 8x8

        # 卷积层3-5: 保持8x8尺寸
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 8x8 → 4x4

        # 全连接层
        nn.Flatten(),
        nn.Linear(256*4*4, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 100)  # 100类输出
    )
按7:3比例划分训练集/验证集
train_images, val_images = train_test_split(images, train_size=0.7)

生成train.txt/val.txt格式：
path/to/image1.jpg 0
path/to/image2.jpg 1
class ImageTxtDataset(data.Dataset): def init(self, txt_path, img_root, transform=None): # 从txt文件读取图像路径和标签 self.imgs_path = [] self.labels = []

def __getitem__(self, index):
    img = Image.open(path).convert("RGB")
    if self.transform:
        img = self.transform(img)
    return img, label
数据增强配置
train_transform = transforms.Compose([ transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(), # 随机水平翻转 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

训练循环
for epoch in range(epochs): model.train() for imgs, targets in train_loader: # 前向传播 outputs = model(imgs) loss = loss_fn(outputs, targets)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 验证评估
model.eval()
with torch.no_grad():
    total_correct = (outputs.argmax(1) == targets).sum().item()

# 保存模型
torch.save(model, f"alex_{epoch}.pth")
预处理管道
transform = transforms.Compose([ transforms.Resize((32, 32)), transforms.ToTensor() ])

预测流程
model = torch.load("model_save/alex_9.pth") model.eval() # 切换到评估模式

with torch.no_grad(): output = model(input_tensor) predicted_class = output.argmax(1).item()

