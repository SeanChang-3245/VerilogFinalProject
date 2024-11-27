import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from VAE import ConvVAE, loss_function, save_model

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

# Validation function
def validate(model, val_loader, beta=1.0):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar, beta).item()
    average_loss = val_loss / len(val_loader.dataset)
    print(f'====> Validation loss: {average_loss:.4f}')
    return average_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop with early stopping
    loss_values = []
    val_loss_values = []
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch, loss_values, beta=1)
        val_loss = validate(model, val_loader, beta=1)
        val_loss_values.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, 'vae_conv_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

    # Save the final model
    save_model(model, 'vae_conv_final.pth')