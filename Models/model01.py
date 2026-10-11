import torch
from torch import nn

class Contrail_CNN(nn.Module):
    
    """
    U-net style architecture.
      - Encoder: 4 sequential blocks of Conv2d -> BatchNorm2d -> ReLU,
        each followed by 2x2 MaxPool2d downsampling (channels: input -> 64 -> 128 -> 256 -> 512).
      - Bottleneck: Conv2d -> BatchNorm2d -> ReLU maintaining the 16x16 feature map at 512 channels.
      - Decoder: 4 sequential stages using ConvTranspose2d (2x stride) paired with channel concatenation of encoder skip-connections.
      - Output Head: 1x1 Conv2d mapping the final 64 feature maps to `output_channels` (raw logits).

    Shape (default 256x256 input):
      - Input:  (Batch, input_channels, 256, 256)
      - Output: (Batch, output_channels, 256, 256)
    """

    def __init__(self, input_channels=3, output_channels=1,                 conv_kernel_size=3, pool_kernel_size=2, pool_stride=2):
        super().__init__()

        # --- Encoder ---

        # Level 1: 256x256x3 -> 128x128x64
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Level 2: 128x128x64 -> 64x64x128
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Level 3: 64x64x128 -> 32x32x256
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Level 4: 32x32x256 -> 16x16x512
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # --- Bottleneck ---

        # Level 5: 16x16x512 -> 16x16x512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        # Level 4: 16x16x512 -> 32x32x256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=conv_kernel_size, padding="same"), # 256 + 512 to account for concatenation with encoder output
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Level 3: 32x32x256 -> 64x64x128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Level 2: 64x64x128 -> 128x128x64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Level 1: 128x128x64 -> 256x256x64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=pool_kernel_size, stride=pool_stride)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=conv_kernel_size, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Output layer: 256x256x64 -> 256x256x1
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)