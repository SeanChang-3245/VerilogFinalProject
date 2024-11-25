import torch
import torch.nn as nn

class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(JordanRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Input to hidden layer
        self.input_to_hidden = nn.Linear(input_size + output_size, hidden_size)
        # Hidden to output layer
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x, prev_output=None):
        batch_size = x.size(0)
        
        # Initialize previous output if None
        if prev_output is None:
            prev_output = torch.zeros(batch_size, self.hidden_to_output.out_features, 
                                    device=x.device)
        
        # Combine input with previous output
        combined = torch.cat((x, prev_output), dim=1)
        
        # Process through layers
        hidden = self.tanh(self.input_to_hidden(combined))
        output = self.hidden_to_output(hidden)
        
        return output
