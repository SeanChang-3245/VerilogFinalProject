import torch
import torch.nn as nn

class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=25):
        super(ElmanRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Replace linear layers with Conv1d
        self.input_to_hidden = nn.Conv1d(
            in_channels=input_size + hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.hidden_to_output = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=output_size,
            kernel_size=1
        )
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden):
        # Reshape inputs for Conv1d [batch, channels, length]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)  # [batch, 1, 1]
        hidden = hidden.view(batch_size, -1, 1)  # [batch, hidden_size, 1]
        combined = torch.cat((x, hidden), dim=1)  # [batch, 1+hidden_size, 1]
        
        # Apply convolutions
        hidden = self.tanh(self.input_to_hidden(combined))
        output = self.hidden_to_output(hidden)
        
        # Squeeze back to original dimensions [batch, output_size]
        return output.squeeze(-1), hidden.squeeze(-1)
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size).to(next(self.parameters()).device)
