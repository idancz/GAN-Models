import numpy as np
import torch
import torch.nn as nn

from math import ceil
import matplotlib.pyplot as plt


latent_size = 128 #from paper
"""**Seed**"""

torch.manual_seed(0)
np.random.seed(0)
seed=0

"""**GPU**"""

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
"""# **WGAN GP**

**Residual Block**
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=latent_size, IsGenerator=True, WithPooling=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.IsGenerator = IsGenerator
        self.WithPooling = WithPooling
        if IsGenerator == True :
            self.blocks = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3 ,padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3 ,padding=1),
                nn.BatchNorm2d(self.out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3 ,padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        else :  # discriminator
            if WithPooling == True:
                self.blocks = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3 ,padding=1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2),
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3 ,padding=1),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                    nn.AvgPool2d(kernel_size=2)
                )
            else :
                self.blocks = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3 ,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3 ,padding=1),
                )
                self.shortcut = nn.Identity()

        self.activate = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


"""**Generator**"""
class Generator_WGANGP(nn.Module):
    def __init__(self ,latent_size, image_size):
        super().__init__()
        self.input_size = latent_size
        self.output_size = image_size
        self.initial_weight = 0.1
        #### Layers ####
        self.latent_to_features = nn.Sequential(  # latent_size - 128.
            nn.Linear(self.input_size, self.input_size * 4 *4),
            nn.ReLU()
        )
        self.Res_Block = ResidualBlock(self.input_size ,self.input_size ,IsGenerator=True ,WithPooling=False)
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3 ,padding=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.InitParameters()

    def forward(self, noise):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(noise)  # 128*4*4 vector
        # Reshape to [128,4,4]
        x = x.view(-1, self.input_size, 4, 4)  # 128,4,4
        x = self.Res_Block(x)  # 128,8,8
        x = x[:, :, :7, :7]
        x = self.Res_Block(x)  # 128,14,14
        x = self.Res_Block(x)  # 128,28,28
        x = self.output_conv(x)  # 1,28,28
        # Return generated image
        return x

    def InitParameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.initial_weight, self.initial_weight)

    def GenerateNoise(self, num_samples):
        return torch.randn((num_samples, self.input_size))

    def GenerateNewImage(self, num_of_images=2):
        new_noise = self.GenerateNoise(num_of_images).to(device)
        fake_images = self.forward(new_noise)
        num_rows = max(num_of_images // 10, 1)
        num_cols = ceil(num_of_images / num_rows)
        for plot in range(1, num_of_images + 1):
            axes = plt.subplot(num_rows, num_cols, plot)   # create a new subplot
            plt.imshow(fake_images.data[plot - 1].cpu().numpy()[0, :, :], cmap="gray")

"""**Discriminator**"""

class Discriminator_WGANGP(nn.Module):
    def __init__(self, channels_in ,channels_out):
        super(Discriminator_WGANGP, self).__init__()

        self.channels_in = channels_in,
        self.channels_out = channels_out,
        self.initial_weight = 0.1
        self.Res_Block1 = ResidualBlock(1 ,128 ,IsGenerator=False ,WithPooling=True)
        self.Res_Block2 = ResidualBlock(128 ,128 ,IsGenerator=False ,WithPooling=True)
        self.Res_Block3 = ResidualBlock(128 ,128 ,IsGenerator=False ,WithPooling=False)
        self.AvgPool2d  = nn.AvgPool2d(kernel_size=7)

        self.FC = nn.Linear(128, 1)

        self.InitParameters()

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)  # 1,28,28
        out = self.Res_Block1(input)  # 128,14,14
        out = self.Res_Block2(out)  # 128,7,7
        out = self.Res_Block3(out)  # 128,7,7
        out = self.Res_Block3(out)  # 128,7,7
        out = self.AvgPool2d(out)
        out = out.view(-1 ,128)  # 128,1,1
        out = self.FC(out) # 1

        return out.view(-1)

    def InitParameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.initial_weight, self.initial_weight)


"""# **DCGAN**

**Generator**
"""


class Generator_DCGAN(nn.Module):
    def __init__(self, latent_size, dim):
        super().__init__()
        self.input_size = latent_size
        self.dim = dim
        self.initial_weight = 0.1
        #### Layers ####
        self.latent_to_features = nn.Sequential(  # latent_size - 128.
            nn.Linear(self.input_size, self.dim * 4 * 4 * 4),  # according to paper implementation
            nn.ReLU()
        )
        self.block_1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * self.dim),
        )
        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.dim, self.dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(self.dim),
        )
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(self.dim, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.InitParameters()

    def forward(self, noise):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(noise)
        # Reshape
        x = x.view(-1, 4 * self.dim, 4, 4)
        # Apply first residual block
        x = self.block_1(x)
        x = x[:, :, :7, :7]
        # Apply second residual block
        x = self.block_2(x)
        x = self.output_conv(x)
        # Return generated image
        return x

    def InitParameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.initial_weight, self.initial_weight)

    def GenerateNoise(self, num_samples):
        return torch.randn((num_samples, self.input_size))

    def GenerateNewImage(self, num_of_images=2):
        new_noise = self.GenerateNoise(num_of_images).to(device)
        fake_images = self.forward(new_noise)
        num_rows = max(num_of_images // 10, 1)
        num_cols = ceil(num_of_images / num_rows)
        for plot in range(1, num_of_images + 1):
            axes = plt.subplot(num_rows, num_cols, plot)  # create a new subplot
            plt.imshow(fake_images.data[plot - 1].cpu().numpy()[0, :, :], cmap="gray")


"""**Discriminator**"""


class Discriminator_DCGAN(nn.Module):

    def __init__(self, dim=16):
        super(Discriminator_DCGAN, self).__init__()
        self.dim = dim
        self.initial_weight = 0.1

        self.main = nn.Sequential(
            nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(self.dim, 2 * self.dim, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(2 * self.dim, 4 * self.dim, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.output = nn.Sequential(
            nn.Linear(4 * 4 * 4 * self.dim, 1),
            nn.Sigmoid()
        )

        self.InitParameters()

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * self.dim)
        out = self.output(out)
        return out.view(-1)

    def InitParameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.initial_weight, self.initial_weight)

