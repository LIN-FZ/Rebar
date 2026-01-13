import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgBoost(nn.Module):
    def __init__(self):
        super(ImgBoost, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.deblur_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.deblur_conv2 = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.noise_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.noise_batchnorm = nn.BatchNorm2d(3)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

        self.attention = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x_deblur = self.leaky_relu(self.deblur_conv1(x))
        x_deblur = self.relu(self.deblur_conv2(x_deblur))

        x_noise = self.noise_batchnorm(self.leaky_relu(self.noise_conv1(x)))

        x1 = self.relu(self.e_conv1(x_noise))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))

        attention_mask = self.attention(x5)
        enhanced_feature = x5 * attention_mask

        clean_image = self.relu((enhanced_feature * x_noise) - enhanced_feature + 1)

        enhanced_image = self.upsample(clean_image)

        return enhanced_image

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 480) 
    image = torch.rand(*image_size)

    model = ImgBoost()
    output = model(image)
    print(output.size())