图像分类模型学习笔记
模型架构分析
1. 改进版AlexNet (model.py)
class AlexNet(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.5):
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate/4),
            # ... 4个类似卷积块
            nn.AdaptiveAvgPool2d((6, 6))  # 自适应池化

        # 分类器层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128*6*6, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # ... 2层全连接
        )

        # 权重初始化
        self._initialize_weights()
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        # Patch嵌入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

        # CLS Token + 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
ViT(
    image_size=256,    # 输入尺寸
    patch_size=16,     # Patch大小
    num_classes=100,   # 输出类别
    dim=1024,          # 嵌入维度
    depth=6,           # Transformer层数
    heads=16,          # 注意力头数
    mlp_dim=2048       # MLP隐藏层维度
)
# 图像预处理
transform = transforms.Compose([
    transforms.Resize(224),                # 调整尺寸
    transforms.RandomHorizontalFlip(),     # 数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集加载
train_data = ImageTxtDataset(txt_path, img_root, transform)
test_data = ImageTxtDataset(txt_path, img_root, transform)
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)
for epoch in range(epochs):
    # 训练阶段
    model.train()
    for imgs, targets in train_loader:
        # 数据验证
        if (targets < 0).any() or (targets >= 100).any():
            continue  # 跳过无效标签

        # 前向传播
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 反向传播
        loss.backward()
        optim.step()

    # 学习率更新
    scheduler.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        for imgs, targets in test_loader:
            outputs = model(imgs)
            # 计算准确率

    # 模型保存
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), "best_model.pth")