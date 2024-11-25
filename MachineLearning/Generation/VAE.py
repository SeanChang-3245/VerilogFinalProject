import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallLatentAudioGenerator(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.target_length = 12000

        # Encoder with new kernel size and stride
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=5, padding=2),  # 2396
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=25, stride=5, padding=2),  # 476
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=25, stride=5, padding=2),  # 92
            nn.ReLU(),
            nn.Flatten()
        )

        # Updated dimensions for linear layers
        self.fc_mu = nn.Linear(32 * 92, latent_dim)
        self.fc_var = nn.Linear(32 * 92, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 32 * 92)
        
        # Decoder with matching kernel size and stride
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=25, stride=5, padding=2),  # 476
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=25, stride=5, padding=2),  # 2396
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=25, stride=5, padding=0),   # 12000
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
        x = x.view(-1, 32, 92)  # Updated dimension
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var