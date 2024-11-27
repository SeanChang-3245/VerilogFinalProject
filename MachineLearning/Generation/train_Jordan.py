import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from JordanRNN import JordanRNN
from audio_preprocess.audio_dataset import AudioDataset, JordanDataset

# Hyperparameters
learning_rate = 0.001
num_epochs = 100
output_size = 200  # Example output size, adjust as needed
batch_size = 8
sequence_len = 12000
audio_dir = 'F:\\Sean\\VerilogWithML\\downsample'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = JordanDataset(audio_dir=audio_dir, seq_length=sequence_len, device=device)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

# Model, loss function, optimizer
model = JordanRNN(output_size, sequence_len).to(device) 
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []

# Training loop
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1:3}/{num_epochs}], Loss: {loss.item():.6f}')


# Save the model
model_save_path = './checkpoints/jordan_rnn_checkpoint.pt'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Plot the training losses
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()
print("Training complete.")
