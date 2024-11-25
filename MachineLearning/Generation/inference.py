import torch
import numpy as np
import scipy.io.wavfile as wav
from RNN import ElmanRNN

def generate_audio(model, length=12000, temperature=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        generated = torch.zeros(length)
        hidden = model.init_hidden()
        x = torch.randn(1, 1).to(device) * 0.01
        
        for i in range(length):
            # Pass input directly without concatenating
            output, hidden = model(x, hidden)
            
            # Add some randomness with temperature
            if temperature > 0:
                output = output + torch.randn_like(output) * temperature * 0.1
            
            generated[i] = output.item()
            x = output.unsqueeze(0)  # Reshape for next iteration
        
    generated = generated.cpu().numpy()
    generated = generated / np.max(np.abs(generated))
    return generated

def save_wav(filename, data, sr=4000):
    wav.write(filename, sr, (data * 32767).astype(np.int16))

if __name__ == "__main__":
    model = ElmanRNN(input_size=1, hidden_size=32, output_size=1)
    model.load_state_dict(torch.load('./checkpoints/model_epoch_3_loss_0.711402.pth', map_location='cuda'))
    generated_audio = generate_audio(model)
    save_wav('output.wav', generated_audio)