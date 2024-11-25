import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallLatentAudioGenerator(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.target_length = 12000

        # Enhanced Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=5, padding=2),  # 2396
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=2),  # 476
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=25, stride=5, padding=2),  # 92
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        # Updated dimensions for linear layers
        self.fc_mu = nn.Linear(128 * 92, latent_dim)
        self.fc_var = nn.Linear(128 * 92, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 92)
        
        # Enhanced Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=25, stride=5, padding=2),  # 476
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=25, stride=5, padding=2),  # 2396
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=25, stride=5, padding=0),   # 12000
            nn.Tanh()
        )

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 128, 92)  # Updated dimension
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var