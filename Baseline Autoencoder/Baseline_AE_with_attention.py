import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a1 = self.fc(self.avg_pool(x))
        a2 = self.fc(self.max_pool(x))
        return x * self.sigmoid(a1 + a2)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mx, _ = torch.max(x, dim=1, keepdim=True)
        av = torch.mean(x, dim=1, keepdim=True)
        m = torch.cat([mx, av], dim=1)
        return x * self.sigmoid(self.conv(m))

# CBAM Residual Block
class CBAMResBlock(nn.Module):
    def __init__(self, channels, dropout=0.2, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.use_residual else torch.zeros_like(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        if self.use_residual:
            out = out + residual
        return self.relu(out)

class RevisedAutoencoder(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_channels, 1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec1_conv = nn.ConvTranspose2d(latent_channels, 128, 4, 2, 1)
        self.dec1_cbam = CBAMResBlock(128 + 128, dropout=0.2, use_residual=False)
        self.dec1_bn = nn.BatchNorm2d(128)

        self.dec2_conv = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2_cbam = CBAMResBlock(64, dropout=0.2, use_residual=False)
        self.dec2_bn = nn.BatchNorm2d(64)

        self.dec3_conv = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec3_cbam = CBAMResBlock(32, dropout=0.2, use_residual=False)
        self.dec3_bn = nn.BatchNorm2d(32)

        # Final symmetrical upsample
        self.out_conv = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Bottleneck
        z = self.bottleneck(x4)

        # Decoder with one skip (from x3)
        d1 = self.dec1_conv(z)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.dec1_cbam(d1)
        d1 = self.dec1_bn(d1[:, :128, :, :])

        d2 = self.dec2_conv(d1)
        d2 = self.dec2_cbam(d2)
        d2 = self.dec2_bn(d2)

        d3 = self.dec3_conv(d2)
        d3 = self.dec3_cbam(d3)
        d3 = self.dec3_bn(d3)

        out = self.out_conv(d3)
        return self.out_act(out)

# Instantiate the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RevisedAutoencoder(latent_channels=128).to(device)
