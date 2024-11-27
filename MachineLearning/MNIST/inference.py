import torch
from VAE import ConvVAE
import numpy as np

# Inference function with model loading
def generate_images(model_path, num_images, dim1_range, dim2_range, fixed_value=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create a grid for the two dimensions
    param1s = np.linspace(dim1_range[0], dim1_range[1], num_images, dtype=np.float32)
    param2s = np.linspace(dim2_range[0], dim2_range[1], num_images, dtype=np.float32)
    
    # Initialize the latent space with fixed values
    z = torch.full((num_images, 20), fixed_value, dtype=torch.float32).to(device)
    
    # Change the first two dimensions
    z[:, 0] = torch.tensor(param1s, dtype=torch.float32).to(device)
    z[:, 1] = torch.tensor(param2s, dtype=torch.float32).to(device)
    
    samples = []
    for i in range(num_images):
        z_single = z[i].unsqueeze(0)
        with torch.no_grad():
            sample = model.decode(z_single).cpu()
            samples.append(sample)
    samples = torch.cat(samples, dim=0)

    return samples.view(-1, 1, 28, 28)

if __name__ == '__main__':
    num_images = 100
    dim1_range = (-3, 3)
    dim2_range = (-3, 3)
    samples = generate_images('vae_conv_best.pth', num_images, dim1_range, dim2_range)
    
    # Save generated images
    import torchvision.utils as vutils
    vutils.save_image(samples, 'generated_samples.png', nrow=10, normalize=True)