import torch
import torch.nn as nn
import torch.nn.functional as F

class UnblurNet3(nn.Module):
    def __init__(self):
        super(UnblurNet3, self).__init__()

        self.noise_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.noise_batchnorm = nn.BatchNorm2d(3)
        self.leaky_relu = nn.LeakyReLU(0.2)  # 添加缺失的leaky_relu属性

    def forward(self, x):
        # 去噪
        x_noise = self.noise_batchnorm(self.leaky_relu(self.noise_conv1(x)))

        return x_noise

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 480)  # 低分辨率示例
    image = torch.rand(*image_size)

    model = UnblurNet3()
    output = model(image)
    print(output.size())  # 应输出 (1, 3, 640, 480)