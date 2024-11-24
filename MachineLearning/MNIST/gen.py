import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MachineLearning.MNIST.VAE_conv import ConvVAE, loss_function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_images(model_path, num_images, param1_range, param2_range):
    model = ConvVAE().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    param1_values = torch.linspace(param1_range[0], param1_range[1], num_images)
    param2_values = torch.linspace(param2_range[0], param2_range[1], num_images)
    
    samples = []
    with torch.no_grad():
        z = torch.randn(1, 20).to(device)  # Initialize latent vector with zeros
        for param1 in param1_values:
            for param2 in param2_values:
                z[0, 0] = param1  # Change the first parameter
                z[0, 3] = param2  # Change the second parameter
                sample = model.decode(z).cpu()
                samples.append(sample.view(1, 1, 28, 28))
    
    samples = torch.cat(samples)
    return samples

# Define the range for the two parameters
param1_range = (-5, 5)
param2_range = (-5, 5)

# Generate images
samples = generate_images('./vae_conv.pth', 10, param1_range, param2_range)

# Save generated images
import torchvision.utils as vutils
vutils.save_image(samples, 'generated_samples.png', nrow=10, normalize=True)