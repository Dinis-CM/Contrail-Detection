import torch
from torch import nn


class Contrail_ResUNet(nn.Module):
    """
    Residual U-Net architecture with single-conv blocks.
    - Encoder: 4 stages of single Conv2d -> BatchNorm2d added with a 1x1 shortcut projection,
      followed by ReLU and 2x2 MaxPool2d downsampling (channels: input -> 64 -> 128 -> 256 -> 512).
    - Bottleneck: Single Conv2d -> BatchNorm2d + identity shortcut, followed by ReLU.
    - Decoder: 4 sequential stages using ConvTranspose2d (2x stride) paired with channel concatenation
      of encoder skip-connections, followed by a single Conv2d -> BatchNorm2d + 1x1 shortcut, and ReLU.
    - Output Head: 1x1 Conv2d mapping the final 64 feature maps to `output_channels` (raw logits).

    Shape (default 256x256 input):
    - Input:  (Batch, input_channels, 256, 256)
    - Output: (Batch, output_channels, 256, 256)
    """

    def __init__(
        self,
        input_channels=3,
        output_channels=1,
        conv_kernel_size=3,
        pool_kernel_size=2,
        pool_stride=2,
    ):
        super().__init__()

        # --- Encoder ---

        # Level 1: 256x256x3 -> 128x128x64
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(64),
        )
        self.enc1_skip = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.enc1_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Level 2: 128x128x64 -> 64x64x128
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(128),
        )
        self.enc2_skip = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.enc2_relu = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Level 3: 64x64x128 -> 32x32x256
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(256),
        )
        self.enc3_skip = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.enc3_relu = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Level 4: 32x32x256 -> 16x16x512
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(512),
        )
        self.enc4_skip = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.enc4_relu = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # --- Bottleneck ---

        # Level 5: 16x16x512 -> 16x16x512 (Channels identical, pure identity shortcut)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(512),
        )
        self.bottleneck_relu = nn.ReLU(inplace=True)

        # --- Decoder ---

        # Level 4: 16x16x512 -> 32x32x256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(256),
        )
        self.dec4_skip = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.dec4_relu = nn.ReLU(inplace=True)

        # Level 3: 32x32x256 -> 64x64x128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(128),
        )
        self.dec3_skip = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.dec3_relu = nn.ReLU(inplace=True)

        # Level 2: 64x64x128 -> 128x128x64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(64),
        )
        self.dec2_skip = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.dec2_relu = nn.ReLU(inplace=True)

        # Level 1: 128x128x64 -> 256x256x64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=conv_kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(64),
        )
        self.dec1_skip = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.dec1_relu = nn.ReLU(inplace=True)

        # Output layer: 256x256x64 -> 256x256x1
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1_relu(self.enc1_conv(x) + self.enc1_skip(x))
        p1 = self.pool1(e1)

        e2 = self.enc2_relu(self.enc2_conv(p1) + self.enc2_skip(p1))
        p2 = self.pool2(e2)

        e3 = self.enc3_relu(self.enc3_conv(p2) + self.enc3_skip(p2))
        p3 = self.pool3(e3)

        e4 = self.enc4_relu(self.enc4_conv(p3) + self.enc4_skip(p3))
        p4 = self.pool4(e4)

        # Bottleneck (Identity shortcut because channels remain 512 -> 512)
        b = self.bottleneck_relu(self.bottleneck_conv(p4) + p4)

        # Decoder
        cat4 = torch.cat([self.up4(b), e4], dim=1)
        d4 = self.dec4_relu(self.dec4_conv(cat4) + self.dec4_skip(cat4))

        cat3 = torch.cat([self.up3(d4), e3], dim=1)
        d3 = self.dec3_relu(self.dec3_conv(cat3) + self.dec3_skip(cat3))

        cat2 = torch.cat([self.up2(d3), e2], dim=1)
        d2 = self.dec2_relu(self.dec2_conv(cat2) + self.dec2_skip(cat2))

        cat1 = torch.cat([self.up1(d2), e1], dim=1)
        d1 = self.dec1_relu(self.dec1_conv(cat1) + self.dec1_skip(cat1))

        return self.out_conv(d1)