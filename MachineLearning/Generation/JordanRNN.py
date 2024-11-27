import torch
import torch.nn as nn

class JordanRNN(nn.Module):
    def __init__(self, output_size, sequence_len):
        super(JordanRNN, self).__init__()
        
        self.sequence_len = sequence_len
        self.output_size = output_size
        self.input_size = 2*output_size
        
        self.down_conv1 = nn.Conv1d(in_channels=1,  # 400 -> 94
                                    out_channels=4, 
                                    kernel_size=25,
                                    stride=4)

        self.down_conv2 = nn.Conv1d(in_channels=4, # 94 -> 18
                                    out_channels=16, 
                                    kernel_size=25,
                                    stride=4)
        
        self.linear = nn.Linear(16*18, self.output_size)
        
        self.tanh = nn.Tanh()
    
    def forward_once(self, x, prev_output=None):
        batch_size = x.size(0)
        
        # Initialize previous output if None
        if prev_output is None:
            prev_output = torch.zeros(batch_size, self.output_size, 
                                    device=x.device)
        
        # Combine input with previous output
        combined = torch.cat((x, prev_output), dim=1)
        
        # Process through layers
        x = self.down_conv1(combined.unsqueeze(1))
        x = self.tanh(x)
        
        x = self.down_conv2(x)
        x = self.tanh(x)
        
        x = x.view(batch_size, -1)
        output = self.linear(x)
        
        return output
        
        
    def forward(self, x):
        prev_output = None
        outputs = []
        
        if self.sequence_len % self.output_size != 0:
            raise ValueError("Sequence length must be divisible by output size.")
        
        for i in range(self.sequence_len // self.output_size):
            output = self.forward_once(x, prev_output)
            outputs.append(output)
            prev_output = output
        
        return torch.cat(outputs, dim=1)
