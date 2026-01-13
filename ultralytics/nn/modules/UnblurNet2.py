import torch
import torch.nn as nn
import torch.nn.functional as F


class UnblurNet2(nn.Module):
    def __init__(self):
        super(UnblurNet2, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 上采样提升分辨率
        enhanced_image = self.upsample(x)

        return enhanced_image

if __name__ == "__main__":
    # 生成示例图像
    image_size = (1, 3, 640, 480)  # 低分辨率示例
    image = torch.rand(*image_size)

    model = UnblurNet2()
    output = model(image)
    print(output.size())  # 应输出 (1, 3, 1280, 960)