import torch
import numpy as np
import soundfile as sf
from JordanRNN import JordanRNN

def inference(model_path, sequence_len, output_size, output_audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = JordanRNN(output_size, sequence_len).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate a random input vector
    random_input = torch.randn(1, output_size, device=device)
    
    # Perform inference
    with torch.no_grad():
        output = model(random_input)
    
    # Convert output to numpy array
    output_np = output.cpu().numpy().flatten()
    
    # Normalize the output to be in the range of -1.0 to 1.0
    output_np = output_np / np.max(np.abs(output_np))
    
    # Save the output as an audio file
    sf.write(output_audio_path, output_np, samplerate=4000)
    
    return output_np

if __name__ == "__main__":
    model_path = './checkpoints/jordan_rnn_checkpoint.pt'
    sequence_len = 12000
    output_size = 200
    output_audio_path = './output_audio.wav'
    
    output = inference(model_path, sequence_len, output_size, output_audio_path)
    print("Inference output saved to:", output_audio_path)
