# train_alex.py
import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from alex import Alex  # 导入Alex模型

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="./dataset_chen",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="./dataset_chen",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
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