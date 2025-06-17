# test_alex.py
import torch
import torchvision
from PIL import Image
from alex import Alex  # 导入Alex模型
from torchvision import transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载图像
image_path = "your_image.jpg"  # 替换为你的图像路径
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度

# 加载模型
model = torch.load("model_save/alex_9.pth", map_location=device)
model.eval()  # 设置为评估模式

# 预测
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(1).item()

# CIFAR-10类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"预测结果: {classes[predicted_class]}")