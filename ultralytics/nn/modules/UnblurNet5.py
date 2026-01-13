import torch
import torch.nn as nn
import torch.nn.functional as F

class UnblurNet5(nn.Module):
    def __init__(self):
        super(UnblurNet5, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 去模糊模块
        self.deblur_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.deblur_conv2 = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1, bias=True)

        # 去噪模块
        self.noise_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.noise_batchnorm = nn.BatchNorm2d(3)

        # 上采样模块
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 去模糊
        x_deblur = self.leaky_relu(self.deblur_conv1(x))
        x_deblur = self.relu(self.deblur_conv2(x_deblur))

        # 去噪
        x_noise = self.noise_batchnorm(self.leaky_relu(self.noise_conv1(x)))

        # 上采样提升分辨率
        enhanced_image = self.upsample(x_noise)

        return enhanced_image

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 480)  # 低分辨率示例
    image = torch.rand(*image_size)

    model = UnblurNet5()
    output = model(image)
    print(output.size())  # 应输出 (1, 3, 1280, 960)