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
        self.enc_conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)  # 28x28 -> 14x14, further reduced filters
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7, further reduced filters
        self.enc_fc1 = nn.Linear(16 * 7 * 7, 64)  # further reduced size
        self.enc_fc2_mu = nn.Linear(64, 20)  # Latent dimension remains 20
        self.enc_fc2_logvar = nn.Linear(64, 20)  # Latent dimension remains 20
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        
        # Decoder
        self.dec_fc1 = nn.Linear(20, 32)  # further reduced size
        self.dec_fc2 = nn.Linear(32, 8 * 7 * 7)  # further reduced size
        self.dec_conv1 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14, further reduced filters
        self.dec_conv2 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28, further reduced filters
        
        # Activation function
        self.relu = nn.ReLU()   
        
    def encode(self, x):
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = h.view(-1, 16 * 7 * 7)
        h = self.dropout(self.relu(self.enc_fc1(h)))  # Apply dropout
        return self.enc_fc2_mu(h), self.enc_fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        h = h.view(-1, 8, 7, 7)
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

# Training function with loss tracking
def train(model, train_loader, optimizer, epoch, loss_values, beta=1.0):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
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


# Inference function with model loading
def generate_images(model_path, num_images):
    model = ConvVAE().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 20).to(device)  # Latent dimension changed to 20
        samples = model.decode(z).cpu()
        return samples.view(-1, 1, 28, 28)

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

if __name__ == '__main__':
    # Main script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    loss_values = []
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch, loss_values, beta=1.0)

    # Save the model
    save_model(model, 'vae_conv.pth')

    # Generate images
    num_images = 100
    samples = generate_images('vae_conv.pth', num_images)
    # Save generated images
    import torchvision.utils as vutils
    vutils.save_image(samples, 'generated_samples.png', nrow=4, normalize=True)