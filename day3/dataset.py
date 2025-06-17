import os
from torch.utils import data
from PIL import Image


class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, img_root: str, transform=None):  # 参数名改为img_root更清晰
        self.transform = transform
        self.img_root = img_root  # 存储图像根目录
        self.imgs_path = []
        self.labels = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # 跳过格式错误行

            img_name = parts[0]  # 图像文件名
            label = int(parts[1])  # 标签

            # 正确拼接完整路径
            img_path = os.path.join(self.img_root, img_name)
            self.imgs_path.append(img_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        img_path = self.imgs_path[i]
        label = self.labels[i]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法打开图像: {img_path}, 错误: {e}")
            # 返回空白图像作为占位符
            image = Image.new('RGB', (32, 32), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label