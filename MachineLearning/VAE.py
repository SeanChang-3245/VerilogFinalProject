import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(784, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_fc3_mu = nn.Linear(256, 32)  # Latent dimension changed to 32
        self.enc_fc3_logvar = nn.Linear(256, 32)  # Latent dimension changed to 32
        
        # Decoder
        self.dec_fc1 = nn.Linear(32, 256)  # Latent dimension changed to 32
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_fc3 = nn.Linear(512, 784)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def encode(self, x):
        h = self.relu(self.enc_fc1(x))
        h = self.relu(self.enc_fc2(h))
        return self.enc_fc3_mu(h), self.enc_fc3_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_fc3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# Training function with loss tracking
def train(model, train_loader, optimizer, epoch, loss_values, beta=1.0):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    average_loss = train_loss / len(train_loader.dataset)
    loss_values.append(average_loss)
    print(f'====> Epoch: {epoch} Average loss: {average_loss:.4f}')


# Inference function
def generate_images(model, num_images):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 32).to(device)
        samples = model.decode(z).cpu()
        return samples.view(-1, 1, 28, 28)

if __name__ == '__main__':
    # Main script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    loss_values = []
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch, loss_values, beta=1.0)

    # Generate images
    num_images = 16
    samples = generate_images(model, num_images)
    # Save generated images
    import torchvision.utils as vutils
    vutils.save_image(samples, 'generated_samples.png', nrow=4, normalize=True)