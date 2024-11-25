import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from VAE import SmallLatentAudioGenerator
from audio_dataset import AudioDataset
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x.squeeze(), x)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss

def train_vae(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = vae_loss(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item()/len(data):.6f}')
    
    return train_loss / len(train_loader.dataset)

def main():
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    epochs = 100
    learning_rate = 5e-4
    latent_dim = 32
    seq_length = 12000  # Match with VAE's target_length
    
    # Initialize model and optimizer
    model = SmallLatentAudioGenerator(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Data loading using the existing AudioDataset
    dataset = AudioDataset("F:\\Sean\\VerilogWithML\\downsample", seq_length=seq_length, device=device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train_vae(model, train_loader, optimizer, device, epoch)
        losses.append(loss)
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'vae_checkpoint_epoch_{epoch}.pt')
            
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()

if __name__ == "__main__":
    main()
