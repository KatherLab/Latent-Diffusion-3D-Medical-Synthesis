import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock3D, self).__init__()

        conv_block = [  nn.ReflectionPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        nn.InstanceNorm3d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        nn.InstanceNorm3d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator3D(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):
        super(Generator3D, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad3d(3),
                    nn.Conv3d(input_nc, 64, 7),
                    nn.InstanceNorm3d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock3D(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad3d(3),
                    nn.Conv3d(64, output_nc, 7),
                    ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator3D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator3D, self).__init__()

        model = [   nn.Conv3d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(256, 512, 4, padding=1),
                    nn.InstanceNorm3d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv3d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)

# ---------- TEST CODE ----------
if __name__ == "__main__":
    # Dummy 3D input: (batch_size, channels, depth, height, width)
    x = torch.randn(2, 1, 32, 64, 64)  # Example input

    # Create generator and discriminator
    G = Generator3D(input_nc=1, output_nc=1)
    D = Discriminator3D(input_nc=1)

    # Pass through generator
    fake_x = G(x)
    print("Generator output shape:", fake_x.shape)

    # Pass through discriminator
    pred = D(fake_x)
    print("Discriminator output shape:", pred.shape)
