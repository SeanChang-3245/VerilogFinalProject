import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(64 * 7 * 7, 256)
        self.enc_fc2_mu = nn.Linear(256, 20)
        self.enc_fc2_logvar = nn.Linear(256, 20)
        
        # Decoder
        self.dec_fc1 = nn.Linear(20, 256)
        self.dec_fc2 = nn.Linear(256, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        
    def encode(self, x):
        h = torch.relu(self.enc_conv1(x))
        h = torch.relu(self.enc_conv2(h))
        h = h.view(-1, 64 * 7 * 7)
        h = torch.relu(self.enc_fc1(h))
        return self.enc_fc2_mu(h), self.enc_fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        h = h.view(-1, 64, 7, 7)
        h = torch.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Training function
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Inference function
def generate_images(model, num_images):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 20).to(device)
        samples = model.decode(z).cpu()
        return samples

if __name__ == '__main__':
    # Main script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch)

    # Generate images
    num_images = 16
    samples = generate_images(model, num_images)