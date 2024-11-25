import torch
import numpy as np
from VAE import SmallLatentAudioGenerator

def load_model(model_path, latent_dim=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = SmallLatentAudioGenerator(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    model_path = "vae_checkpoint_epoch_90.pt"
    model = load_model(model_path)
    
    # Generate a single audio sample
    audio_sample = generate_audio(model, num_samples=1)
    
    # Generate interpolation between two random latent vectors
    z1 = np.random.randn(8)  # assuming latent_dim=8
    z2 = np.random.randn(8)
    interpolated = interpolate_latent(z1, z2, steps=10)
    interpolated_audio = generate_audio_from_latent(model, interpolated)
