import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the VAE model with convolutional layers
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # 28x28 -> 14x14, further reduced filters
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 14x14 -> 7x7, further reduced filters
        self.enc_fc1 = nn.Linear(64 * 7 * 7, 64)  # further reduced size
        self.enc_fc2_mu = nn.Linear(64, 16)  # Latent dimension remains 20
        self.enc_fc2_logvar = nn.Linear(64, 16)  # Latent dimension remains 20
        
        # Decoder
        self.dec_fc1 = nn.Linear(16, 32)  # further reduced size
        self.dec_fc2 = nn.Linear(32, 64 * 7 * 7)  # further reduced size
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)  # 7x7 -> 14x14, further reduced filters
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)  # 14x14 -> 28x28, further reduced filters
        
        # Activation function
        self.relu = nn.ReLU()   
        
    def encode(self, x):
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = h.view(-1, 64 * 7 * 7)
        h = self.relu(self.enc_fc1(h))
        return self.enc_fc2_mu(h), self.enc_fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        h = h.view(-1, 64, 7, 7)
        h = self.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')
