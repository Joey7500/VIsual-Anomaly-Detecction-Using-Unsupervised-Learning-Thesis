import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional CBAM block
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid(avg_out + max_out)
        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x

# Main model
class DFR_FeatureCAE(nn.Module):
    def __init__(self, in_channels=832, latent_dim=260, is_bn=True, use_cbam=True, dropout=0.0, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.use_cbam = use_cbam
        
        mid1 = (in_channels + 2 * latent_dim) // 2

        def block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 1)]
            if is_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # Encoder
        self.enc1 = block(in_channels, mid1)
        self.enc2 = block(mid1, 2 * latent_dim)
        self.enc3 = nn.Conv2d(2 * latent_dim, latent_dim, 1)

        # Decoder
        self.dec1 = block(latent_dim, 2 * latent_dim)
        self.dec2 = block(2 * latent_dim, mid1)
        self.dec3 = nn.Conv2d(mid1, in_channels, 1)

        if use_cbam:
            self.cbam1 = CBAM(2 * latent_dim)
            self.cbam2 = CBAM(mid1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        latent = self.enc3(x2)

        # Decoder
        d1 = self.dec1(latent)
        if self.use_cbam:
            d1 = self.cbam1(d1)
        if self.use_skip:
            d1 = d1 + x2  # skip connection

        d2 = self.dec2(d1)
        if self.use_cbam:
            d2 = self.cbam2(d2)
        if self.use_skip:
            d2 = d2 + x1  # skip connection

        out = self.dec3(d2)
        return out
