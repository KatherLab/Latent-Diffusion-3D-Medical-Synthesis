

from models.i2imamba.modules_3d import ResCNN3D


import torch 

if __name__ == "__main__":
    # Dummy 3D input: (batch_size, channels, depth, height, width)
    x = torch.randn(2, 2, 64, 64, 64)  # Example input

    model = ResCNN3D(input_dim=2, output_dim=2)
    
    output = model(x)

    print(output.shape)


