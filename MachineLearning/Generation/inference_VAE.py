import torch
import numpy as np
from VAE import SmallLatentAudioGenerator
from scipy.io.wavfile import write

def load_model(model_path, latent_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = SmallLatentAudioGenerator(latent_dim=latent_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def generate_audio(model, num_samples=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Generate audio samples from random latent vectors"""
    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn(num_samples, model.fc_mu.out_features).to(device)
        # Generate audio through decoder
        generated = model.decode(z)
        # Move to CPU and convert to numpy
        generated = generated.cpu().numpy()
        # Remove channel dimension and normalize
        generated = generated.squeeze(1)
        return generated

def generate_audio_from_latent(model, latent_vector, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Generate audio from a specific latent vector"""
    with torch.no_grad():
        # Ensure latent vector is a torch tensor
        if not isinstance(latent_vector, torch.Tensor):
            latent_vector = torch.tensor(latent_vector, dtype=torch.float32)
        latent_vector = latent_vector.to(device)
        # Reshape if necessary
        if len(latent_vector.shape) == 1:
            latent_vector = latent_vector.unsqueeze(0)
        # Generate audio through decoder
        generated = model.decode(latent_vector)
        # Move to CPU and convert to numpy
        generated = generated.cpu().numpy()
        # Remove channel dimension
        generated = generated.squeeze(1)
        return generated

def interpolate_latent(start_vector, end_vector, steps=10):
    """Create interpolation between two latent vectors"""
    alphas = np.linspace(0, 1, steps)
    vectors = []
    for alpha in alphas:
        vector = start_vector * (1 - alpha) + end_vector * alpha
        vectors.append(vector)
    return np.array(vectors)

if __name__ == "__main__":
    # Example usage
    model_path = "./checkpoints/vae_checkpoint_epoch_90.pt"
    latent_dim = 32  # Ensure this matches the latent_dim in the model
    model = load_model(model_path, latent_dim=latent_dim)
    
    # Generate a single audio sample
    audio_sample = generate_audio(model, num_samples=1)
    # Save the generated audio sample
    file_name = "generated_audio_sample.wav"
    # Assuming a sample rate of 22050 Hz
    sample_rate = 4000
    # Normalize audio to the range [-1, 1]
    audio_sample = audio_sample / np.max(np.abs(audio_sample))
    # Save using scipy.io.wavfile
    write(file_name, sample_rate, audio_sample[0])
    print(f"Saved {file_name}")
    
    # Generate interpolation between two random latent vectors
    z1 = np.random.randn(latent_dim)  # Use the correct latent_dim
    z2 = np.random.randn(latent_dim)  # Use the correct latent_dim
    interpolated = interpolate_latent(z1, z2, steps=10)
    interpolated_audio = generate_audio_from_latent(model, interpolated)
    # Save the generated audio samples
    for i, audio in enumerate(interpolated_audio):
        file_name = f"interpolated_audio_sample_{i}.wav"
        # Assuming a sample rate of 22050 Hz
        sample_rate = 4000
        # Normalize audio to the range [-1, 1]
        audio = audio / np.max(np.abs(audio))
        # Save using scipy.io.wavfile
        write(file_name, sample_rate, audio)
        print(f"Saved {file_name}")
