# train_alex.py
import time
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from alex import Alex
from dataset import ImageTxtDataset
from torchvision import transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 统一图像预处理（32x32尺寸匹配模型输入）
common_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整为32x32以匹配模型输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 训练集（添加数据增强）
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # 仅训练集使用数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 准备数据集
train_data = ImageTxtDataset(
    txt_path=r"C:\Users\Lenovo\Desktop\dataset\train.txt",
    img_root=r"C:\Users\Lenovo\Desktop\dataset\image2\train",  # 参数名对应修改
    transform=train_transform
)

# 测试集使用相同的预处理（无数据增强）
test_data = ImageTxtDataset(
    txt_path=r"C:\Users\Lenovo\Desktop\dataset\val.txt",  # 添加测试集txt
    img_root=r"C:\Users\Lenovo\Desktop\dataset\image2\val",  # 测试集图像目录
    transform=common_transform
)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集长度: {train_data_size}")
print(f"测试数据集长度: {test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 创建模型
model = Alex().to(device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练参数
total_train_step = 0
total_test_step = 0
epochs = 20

# TensorBoard
writer = SummaryWriter("./logs_alex")

start_time = time.time()

for epoch in range(epochs):
    print(f"----- 第 {epoch + 1}/{epochs} 轮训练开始 -----")

    # 训练模式
    model.train()
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录日志
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练步数: {total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 评估模式
    model.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()

    # 计算指标
    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_correct / test_data_size

    print(f"测试集平均Loss: {avg_test_loss:.4f}, 准确率: {accuracy:.4f}")
    writer.add_scalar("test_loss", avg_test_loss, epoch)
    writer.add_scalar("test_accuracy", accuracy, epoch)

    # 保存模型
    torch.save(model, f"./model_save/alex_{epoch}.pth")
    print(f"模型已保存: alex_{epoch}.pth")

end_time = time.time()
print(f"总训练时间: {end_time - start_time:.2f}秒")
writer.close()