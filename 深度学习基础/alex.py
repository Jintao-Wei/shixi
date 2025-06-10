# alex.py
import torch
from torch import nn


class Alex(nn.Module):
    def __init__(self):
        super(Alex, self).__init__()
        self.model = nn.Sequential(
            # 第一层：保持尺寸不变
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # 第二层：保持尺寸不变
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            # 第三层：保持尺寸不变
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),

            # 第四层：保持尺寸不变
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),

            # 第五层：保持尺寸不变
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4

            # 展平层
            nn.Flatten(),

            # 全连接层
            nn.Linear(256 * 4 * 4, 4096),  # 输入尺寸修正
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # 测试模型输出尺寸
    alex = Alex()
    input = torch.ones((64, 3, 32, 32))
    output = alex(input)
    print(output.shape)  # 应该输出: torch.Size([64, 10])