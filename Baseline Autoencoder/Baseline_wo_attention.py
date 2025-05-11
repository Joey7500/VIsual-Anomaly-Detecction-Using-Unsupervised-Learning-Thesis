import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineAutoencoder(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()
        # --- Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,   32, 4, 2, 1),  # 512→256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32,  64, 4, 2, 1),  # 256→128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # 128→64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128,256, 4, 2, 1),  # 64→32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # --- Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_channels, kernel_size=1),  
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # --- Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, 4, 2, 1),  # 32→64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),              # 64→128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),               # 128→256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64,  32, 4, 2, 1),               # 256→512
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        return self.output(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
baseline_model = BaselineAutoencoder(latent_channels=128).to(device)
model = BaselineAutoencoder(latent_channels=128).to(device)
