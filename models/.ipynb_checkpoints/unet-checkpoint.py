import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.subsampling import SubsamplingLayer

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob = 0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, x):
        return self.layers(x)

class UNetModel(nn.Module):
    """ A U-Net model that uses residual blocks and includes a self-attention mechanism """
    def __init__(self, drop_rate, device, learn_mask, num_channels=32, pool_kernel_size=2):
        super().__init__()
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask)
        self.down1 = ConvBlock(1, num_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.down2 = ConvBlock(num_channels, num_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.bottleneck = ConvBlock(num_channels * 2, num_channels * 4)
        self.up1 = nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=2, stride=2)
        self.up_block1 = ConvBlock(num_channels * 4, num_channels * 2)
        self.up2 = nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=2, stride=2)
        self.up_block2 = ConvBlock(num_channels * 2, num_channels)
        self.final_conv = nn.Conv2d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.subsample(x)
        #Downsampling
        x1 = self.down1(x)
        x = self.pool1(x1)
        x2 = self.down2(x)
        x = self.pool2(x2)

        # Bottelneck and attention
        x = self.bottleneck(x)
        
        # Upsampling
        x = self.up1(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_block1(x)
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_block2(x)
        
        return self.final_conv(x).squeeze(1)
