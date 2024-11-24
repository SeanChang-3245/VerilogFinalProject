import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchaudio.datasets import LIBRISPEECH
import torchaudio.transforms as transforms



def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def generate_audio(model, input_dim, sequence_length, device):
    model.eval()
    with torch.no_grad():
        random_input = torch.randn(1, sequence_length, input_dim).to(device)
        generated_audio = model(random_input)
        return generated_audio.cpu().numpy()

class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ElmanRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        for t in range(seq_len):
            h = self.activation(self.input_to_hidden(x[:, t, :]) + self.hidden_to_hidden(h))
        
        out = self.hidden_to_output(h)
        return out

# Replace AudioRNN with ElmanRNN in the main function
def main():
    # Define the model parameters
    input_dim = 1  # Dimension of the random input
    hidden_dim = 128
    output_dim = 1  # Dimension of the audio output
    sequence_length = 1000  # Length of the generated audio sequence
    num_layers = 1
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ElmanRNN(input_dim, hidden_dim, output_dim, num_layers).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dummy training data
    train_inputs = torch.randn(1000, sequence_length, input_dim)
    train_targets = torch.randn(1000, output_dim)
    train_dataset = TensorDataset(train_inputs, train_targets)

    # Define a transform to convert audio to spectrogram
    transform = transforms.MelSpectrogram()

    # Load the LIBRISPEECH dataset
    train_dataset = LIBRISPEECH(root='./data', url='train-clean-100', download=True)
    train_dataset = [(transform(waveform), sample_rate, label, speaker_id, chapter_id, utterance_id) for waveform, sample_rate, label, speaker_id, chapter_id, utterance_id in train_dataset]

    # Create a DataLoader for the TIMIT dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Generate audio
    generated_audio = generate_audio(model, input_dim, sequence_length, device)
    print(generated_audio)

if __name__ == "__main__":
    main()