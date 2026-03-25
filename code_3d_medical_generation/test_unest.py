


from models.unest.unest import UNEST
import torch 

model = UNEST(in_channels=2, out_channels=2, spatial_dims=3,
               img_size=(96, 96, 96), num_layers=4)

x = torch.randn(2, 2, 96, 96, 96)  # Example input

output = model(x)

print(output.shape)