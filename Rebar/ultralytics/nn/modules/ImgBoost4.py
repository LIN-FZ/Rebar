import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgBoost4(nn.Module):
    def __init__(self):
        super(ImgBoost4, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.deblur_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.deblur_conv2 = nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.noise_conv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.noise_batchnorm = nn.BatchNorm2d(3)

    def forward(self, x):
        x_deblur = self.leaky_relu(self.deblur_conv1(x))
        x_deblur = self.relu(self.deblur_conv2(x_deblur))

        x_noise = self.noise_batchnorm(self.leaky_relu(self.noise_conv1(x_deblur)))

        return x_noise

if __name__ == "__main__":
    image_size = (1, 3, 640, 480)  
    image = torch.rand(*image_size)

    model = ImgBoostt4()
    output = model(image)
    print(output.size())  