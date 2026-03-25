import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock3D(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock3D, self).__init__()
        conv_block = []
        
        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout3d(0.5)]

        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class BottleneckCNN3D(nn.Module):
    def __init__(self):
        super(BottleneckCNN3D, self).__init__()
        use_bias = False
        norm_layer = nn.BatchNorm3d
        
        model = [
            ResnetBlock3D(
                256,
                norm_layer=norm_layer,
                use_dropout=False,
                use_bias=use_bias,
            )
        ]
        self.residual_cnn = nn.Sequential(*model)

    def forward(self, x):
        x = self.residual_cnn(x)
        return x


class ResCNN3D(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(ResCNN3D, self).__init__()
        ngf = 64
        use_bias = True
        norm_layer = nn.BatchNorm3d

        # Encoder 1
        self.encoder_1 = nn.Sequential(
            nn.Conv3d(input_dim, ngf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )

        # Encoder 2
        self.encoder_2 = nn.Sequential(
            nn.Conv3d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )

        # Encoder 3
        self.encoder_3 = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        )

        self.bottlenecks = nn.Sequential(
            *[BottleneckCNN3D() for _ in range(9)]
        )

        # Decoder 1
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )

        # Decoder 2
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )

        # Decoder 3
        self.decoder_3 = nn.Sequential(
            nn.Conv3d(ngf, output_dim, kernel_size=7, padding=3),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        x = self.bottlenecks(x)

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        
        return x