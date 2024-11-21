import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.optim import RMSprop

class PatchGANDiscriminator(nn.Module):
    # input channel = 4 for latent noise, 3 for real img
    def __init__(self, input_channel=4):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channel, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

def init_discriminator(lr=0.0001, b1=0.5, b2=0.999):
    discriminator = PatchGANDiscriminator()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    return discriminator, criterion, optimizer_D

def init_WGAN(unet, lr_G=4e-6, lr_D=0.00005):
    discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(WGANDiscriminator(4))
    critic_optimizer_D = RMSprop(discriminator.parameters(), lr=lr_D)
    critic_optimizer_G = RMSprop(unet, lr=lr_G)

    return discriminator, critic_optimizer_G, critic_optimizer_D

class WGANDiscriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        y = self.output(x)
        return y

class WGAN(nn.Module):
    def __init__(self, hidden_dim):
        super(WGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4*64*64, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


if __name__ == "__main__":
    # Assuming real_img and generator are defined elsewhere in the code
    discriminator, criterion, optimizer_D = init_discriminator()
    batch_size = 4
    real_img = torch.randn(batch_size, 3, 64, 64).cuda()  # Example placeholder for real_img
    generator = lambda: torch.randn(batch_size, 3, 64, 64).cuda()  # Example placeholder for generator

    real_out = discriminator(real_img)
    real_label = Variable(torch.ones(batch_size, 1)).cuda()
    fake_label = Variable(torch.zeros(batch_size, 1)).cuda()  # Corrected to zeros for fake labels
    loss_real_D = criterion(real_out, real_label)
    real_scores = real_out
    fake_img = generator().detach()
    fake_out = discriminator(fake_img)
    loss_fake_D = criterion(fake_out, fake_label)
    fake_scores = fake_out
    loss_D = loss_real_D + loss_fake_D
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
