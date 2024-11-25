import torch
import torch.optim as optim
import torch.nn as nn
from JordanRNN import JordanRNN

# Hyperparameters
learning_rate = 0.001
num_epochs = 100
output_size = 200  # Example output size, adjust as needed
batch_size = 8
sequence_len = 12000

# Dummy dataset (replace with actual dataset)
train_data = torch.randn(10, sequence_len) 
train_labels = torch.randn(10, sequence_len)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=list(zip(train_data, train_labels)),
    batch_size=batch_size,
    shuffle=True
)

# Model, loss function, optimizer
model = JordanRNN(output_size, sequence_len)  
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print("Training complete.")
