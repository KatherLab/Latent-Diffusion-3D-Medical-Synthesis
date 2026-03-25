

from models.cyclegan3d import Generator3D, Discriminator3D

import torch 

if __name__ == "__main__":
    # Dummy 3D input: (batch_size, channels, depth, height, width)
    x = torch.randn(2, 2, 64, 64, 64)  # Example input

    # Create generator and discriminator
    G = Generator3D(input_nc=2, output_nc=2)
    D = Discriminator3D(input_nc=2)

    # Pass through generator
    fake_x = G(x)
    print("Generator output shape:", fake_x.shape)

    # Pass through discriminator
    pred = D(fake_x)
    print("Discriminator output shape:", pred.shape)
