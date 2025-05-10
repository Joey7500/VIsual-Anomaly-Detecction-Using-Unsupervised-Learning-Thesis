import torch
import torch.nn as nn

class DFR_FeatureCAE(nn.Module):
    def __init__(self, in_channels=832, latent_dim=220, is_bn=True):
        super(DFR_FeatureCAE, self).__init__()
        
        # --- Encoder ---
        mid1 = (in_channels + 2 * latent_dim) // 2
        enc_layers = []
        
        enc_layers.append(nn.Conv2d(in_channels, mid1, kernel_size=1, stride=1, padding=0))
        if is_bn:
            enc_layers.append(nn.BatchNorm2d(mid1))
        enc_layers.append(nn.ReLU())
        
        enc_layers.append(nn.Conv2d(mid1, 2 * latent_dim, kernel_size=1, stride=1, padding=0))
        if is_bn:
            enc_layers.append(nn.BatchNorm2d(2 * latent_dim))
        enc_layers.append(nn.ReLU())
        
        enc_layers.append(nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0))
        self.encoder = nn.Sequential(*enc_layers)
        
        # --- Decoder ---
        dec_layers = []

        dec_layers.append(nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0))
        if is_bn:
            dec_layers.append(nn.BatchNorm2d(2 * latent_dim))
        dec_layers.append(nn.ReLU())
        
        dec_layers.append(nn.Conv2d(2 * latent_dim, mid1, kernel_size=1, stride=1, padding=0))
        if is_bn:
            dec_layers.append(nn.BatchNorm2d(mid1))
        dec_layers.append(nn.ReLU())
        
        dec_layers.append(nn.Conv2d(mid1, in_channels, kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
