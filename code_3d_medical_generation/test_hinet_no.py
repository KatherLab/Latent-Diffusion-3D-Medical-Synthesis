
from models.hinet.syn_model import Multi_modal_generator
import torch 

model = Multi_modal_generator(1, 1, 64)

x = torch.randn(2, 2, 64, 64, 64)  # Example input


output = model(x)

print(output.shape)