import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgBoost5(nn.Module):
    def __init__(self):
        super(ImgBoostt5, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.deblur_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.deblur_conv2 = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.noise_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.noise_batchnorm = nn.BatchNorm2d(3)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x_deblur = self.leaky_relu(self.deblur_conv1(x))
        x_deblur = self.relu(self.deblur_conv2(x_deblur))

        x_noise = self.noise_batchnorm(self.leaky_relu(self.noise_conv1(x)))

        enhanced_image = self.upsample(x_noise)

        return enhanced_image

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 480)  
    image = torch.rand(*image_size)

    model = ImgBoost5()
    output = model(image)
    print(output.size())  