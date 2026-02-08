"""
3D UNet for tumor segmentation. Input/output spatial shape is determined by patch_size.
Output: single channel, sigmoid (binary segmentation).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample x1 and concat with skip x2; then conv. in_ch = x1 channels, skip_ch = x2 channels, out_ch = output channels."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = [x2.size(i) - x1.size(i) for i in range(2, 5)]
        x1 = nn.functional.pad(x1, [diff[2] // 2, diff[2] - diff[2] // 2,
                                    diff[1] // 2, diff[1] - diff[1] // 2,
                                    diff[0] // 2, diff[0] - diff[0] // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D UNet for binary segmentation. Input shape (B, 1, D, H, W) with patch_size (D, H, W).
    Output shape (B, 1, D, H, W) with values in [0, 1] (sigmoid).
    """
    def __init__(self, in_channels=1, out_channels=1, patch_size=(128, 128, 64), features=(32, 64, 128, 256)):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # Encoder: first block conv only, then Down (pool+conv)
        self.encoder.append(DoubleConv(in_channels, features[0]))
        in_ch = features[0]
        for f in features[1:]:
            self.encoder.append(Down(in_ch, f))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.pool = nn.MaxPool3d(2)

        # Decoder: Up(x1_ch, skip_ch, out_ch) — x1 from bottleneck/prev, skip from encoder (reversed order)
        self.decoder.append(Up(features[-1] * 2, features[-1], features[-1]))   # 512, skip 256 -> 256
        self.decoder.append(Up(features[-1], features[-2], features[-2]))       # 256, skip 128 -> 128
        self.decoder.append(Up(features[-2], features[-3], features[-3]))       # 128, skip 64 -> 64
        self.decoder.append(Up(features[-3], features[-4], features[-4]))      # 64, skip 32 -> 32

        self.final = nn.Conv3d(features[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            skips.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoder, reversed(skips)):
            x = dec(x, skip)
        x = self.final(x)
        return self.sigmoid(x)


def build_unet(patch_size, in_channels=1, out_channels=1, **kwargs):
    """Build UNet with given patch_size (list or tuple of 3 ints)."""
    if isinstance(patch_size, (list, tuple)):
        patch_size = tuple(int(x) for x in patch_size)
    else:
        patch_size = (128, 128, 64)
    return UNet3D(in_channels=in_channels, out_channels=out_channels, patch_size=patch_size, **kwargs)
