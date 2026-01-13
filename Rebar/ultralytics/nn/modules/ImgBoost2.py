import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgBoost2(nn.Module):
    def __init__(self):
        super(ImgBoost2, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        enhanced_image = self.upsample(x)

        return enhanced_image

if __name__ == "__main__":
    image_size = (1, 3, 640, 480) 
    image = torch.rand(*image_size)

    model = ImgBoost2()
    output = model(image)
    print(output.size())