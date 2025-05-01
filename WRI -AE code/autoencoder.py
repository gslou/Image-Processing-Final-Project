import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import models

class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.freqs = 2.0 ** torch.linspace(0, num_encoding_functions - 1, num_encoding_functions)
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape [batch_size, ... , in_dim]
        Returns:
            encoded: tensor of shape [batch_size, ... , out_dim]
        """
        encoding = []
        if self.include_input:
            encoding.append(x)
            
        # For each frequency, add sin and cos features
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))
                
        # Concatenate all encoded features
        encoded = torch.cat(encoding, dim=-1)
        return encoded

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConditionedEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, pos_dim=2):
        super().__init__()
        
        # Position encoding
        self.position_encoder = PositionalEncoding(num_encoding_functions=6, include_input=True)
        pos_encoded_dim = pos_dim * (1 + 6*2)  # Original + sin/cos at each frequency
        
        # Convert position encoding to feature map
        self.pos_embedding = nn.Sequential(
            nn.Linear(pos_encoded_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
        )
        
        # Encoder layers
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            #nn.AvgPool2d(2),
            ConvBlock(32, 64, stride=2),
            #nn.AvgPool2d(2),
            ConvBlock(64, 128, stride=2),
            #nn.AvgPool2d(2),
            ConvBlock(128, latent_dim, stride=2),
            #nn.AvgPool2d(2),
        )
        
        # Calculate output size after encoder (assuming 64x64 input)
        self.encoder_output_size = 4 * 4 * latent_dim
        
    def forward(self, x, pos):
        # Encode image
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        
        # Reshape to spatial feature map
        self.initial_size = 4
        self.latent_dim = latent_dim
        # Decoder layers with upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
    def forward(self, x):
        #x = x.view(-1, self.latent_dim, self.initial_size, self.initial_size)
        x = self.decoder(x)
        return x

class PositionalAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, pos_dim=2):
        super().__init__()
        self.encoder = ConditionedEncoder(in_channels, latent_dim, pos_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        
    def forward(self, x, pos):
        latent = self.encoder(x, pos)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

class LPIPSWithL2Loss(nn.Module):
    def __init__(self, l2_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        
    def forward(self, output, target):
        l2_loss = F.mse_loss(output, target)
        perceptual_loss = self.lpips_fn(output, target).mean()
        
        total_loss = self.l2_weight * l2_loss + self.perceptual_weight * perceptual_loss
        return total_loss, l2_loss, perceptual_loss