import torch
import torch.nn as nn
import torch.nn.functional as F
from models.subsampling import SubsamplingLayer

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_chans)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop1 = nn.Dropout2d(drop_prob)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_chans)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop2 = nn.Dropout2d(drop_prob)

        # Residual connection
        self.residual = (in_chans == out_chans)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        if self.residual:
            x += residual  # Element-wise addition of input and output
        return x

class UNetModel(nn.Module):
    def __init__(self, drop_rate, device, learn_mask, num_channels=32, pool_kernel_size=2):
        super().__init__()
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask)
        self.down1 = ConvBlock(1, num_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.down2 = ConvBlock(num_channels, num_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.down3 = ConvBlock(num_channels * 2, num_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.bottleneck = ConvBlock(num_channels * 4, num_channels * 8)
        self.up1 = nn.ConvTranspose2d(num_channels * 8, num_channels * 4, kernel_size=2, stride=2)
        self.up_block1 = ConvBlock(num_channels * 8, num_channels * 4)
        self.up2 = nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=2, stride=2)
        self.up_block2 = ConvBlock(num_channels * 4, num_channels * 2)
        self.up3 = nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=2, stride=2)
        self.up_block3 = ConvBlock(num_channels * 2, num_channels)
        self.final_conv = nn.Conv2d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.subsample(x)
        x1 = self.down1(x)
        x = self.pool1(x1)
        x2 = self.down2(x)
        x = self.pool2(x2)
        x3 = self.down3(x)
        x = self.pool3(x3)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_block1(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_block2(x)
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_block3(x)
        return self.final_conv(x).squeeze(1)

