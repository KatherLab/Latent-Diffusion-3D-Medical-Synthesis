


from models.medsyn.train_low_res import Unet3D
import torch 

model = Unet3D(dim=16, channels=2)

x = torch.randn(2, 2, 96, 96, 96)  # Example input

output = model(x)

print(output.shape)