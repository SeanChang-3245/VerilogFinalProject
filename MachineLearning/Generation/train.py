import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RNN import ElmanRNN
from audio_preprocess.audio_dataset import AudioDataset
import os
from heapq import heappush, heappop

def save_checkpoint(model, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, filename)

def train_model(model, audio_dir, num_epochs=100, batch_size=32, seq_length=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize heap for top 5 models (negative loss for max heap)
    best_models = []
    
    model = model.to(device)
    dataset = AudioDataset(audio_dir, seq_length, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size=inputs.size(0))
            loss = 0
            
            for t in range(seq_length):
                input_t = inputs[:, t].unsqueeze(1)  # [batch, 1]
                target_t = targets[:, t].unsqueeze(1)  # [batch, 1]
                output, hidden = model(input_t, hidden.detach())
                loss += criterion(output, target_t)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
        
        # Save model if it's in top 5
        checkpoint_name = f'checkpoints/model_epoch_{epoch}_loss_{avg_loss:.6f}.pth'
        heappush(best_models, (-avg_loss, checkpoint_name))
        save_checkpoint(model, epoch, avg_loss, checkpoint_name)
        
        # Keep only top 5 models
        while len(best_models) > 5:
            _, old_checkpoint = heappop(best_models)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    # Save final model
    torch.save(model.state_dict(), 'rnn_model.pth')
    
    # Print summary of best models
    print("\nTop 5 models:")
    for loss, checkpoint in sorted(best_models):
        print(f"Loss: {-loss:.6f}, Checkpoint: {checkpoint}")

if __name__ == "__main__":
    model = ElmanRNN(input_size=1, hidden_size=32, output_size=1)
    train_model(model, batch_size=32,audio_dir="F:\\Sean\\VerilogWithML\\downsample")
    torch.save(model.state_dict(), 'rnn_model.pth')