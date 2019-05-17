import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random

class residual_block(nn.Module):
    def __init__(self, hidden_dimension, use_bias):
        super(residual_block, self).__init__()
    
        # Residual (Transform)
        self.residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4*hidden_dimension, 4*hidden_dimension, 3 ,1, 0, bias=use_bias),
            nn.InstanceNorm2d(4*hidden_dimension),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4*hidden_dimension, 4*hidden_dimension, 3 ,1, 0, bias=use_bias),
            nn.InstanceNorm2d(4*hidden_dimension),
        )

    def forward(self, x):
        return F.relu(x + self.residual(x))


# Transform image between classes: X -> Y
class Generator(nn.Module):
    def __init__(self, input_dimension, output_dimension, norm=nn.BatchNorm2d, n_residuals=6):
        super(Generator, self).__init__()


        use_bias = norm == nn.InstanceNorm2d
        hidden_dimension = 64
        
        # Make Encoder
        # Layer1
        encoder_layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_dimension, hidden_dimension, 7, 1, 0, bias=use_bias)]
        encoder_layers += [nn.InstanceNorm2d(hidden_dimension), nn.ReLU(inplace=False)] if norm == nn.InstanceNorm2d else [nn.BatchNorm2d(hidden_dimension), nn.ReLU(inplace=False)]
        
        # Layer2
        encoder_layers += [nn.ReflectionPad2d(1), nn.Conv2d(hidden_dimension, 2*hidden_dimension, 3, 2, 0, bias=use_bias)]
        encoder_layers += [nn.InstanceNorm2d(2*hidden_dimension), nn.ReLU(inplace=False)] if norm == nn.InstanceNorm2d else [nn.BatchNorm2d(2*hidden_dimension), nn.ReLU(inplace=False)]
        
        # Layer3
        encoder_layers += [nn.ReflectionPad2d(1), nn.Conv2d(2*hidden_dimension, 4*hidden_dimension, 3, 2, 0, bias=use_bias)]
        encoder_layers += [nn.InstanceNorm2d(4*hidden_dimension), nn.ReLU(inplace=False)] if norm == nn.InstanceNorm2d else [nn.BatchNorm2d(4*hidden_dimension), nn.ReLU(inplace=False)]
        self.encoder = nn.Sequential(*encoder_layers)

        # Make Residual(s)
        res_layers = []
        for _ in range(n_residuals):
            res_layers.append(residual_block(hidden_dimension, use_bias))
        self.residuals = nn.Sequential(*res_layers)

        # Make Decoder
        # L1 - Deconv
        decode_layers = [nn.ConvTranspose2d(4*hidden_dimension, 2*hidden_dimension, 3, 2, 1, output_padding=1, bias=use_bias)]
        decode_layers += [nn.InstanceNorm2d(2*hidden_dimension), nn.ReLU(inplace=False)] if norm == nn.InstanceNorm2d else [nn.BatchNorm2d(2*hidden_dimension), nn.ReLU(inplace=False)]
        
        # L2 - Deconv
        decode_layers += [nn.ConvTranspose2d(2*hidden_dimension, hidden_dimension, 3, 2, 1, output_padding=1, bias=use_bias)]
        decode_layers += [nn.InstanceNorm2d(hidden_dimension), nn.ReLU(inplace=False)] if norm == nn.InstanceNorm2d else [nn.BatchNorm2d(hidden_dimension), nn.ReLU(inplace=False)]
        
        # L3 - Conv
        decode_layers += [nn.ReflectionPad2d(3),
                            nn.Conv2d(hidden_dimension, output_dimension, 7, 1, 0, bias=use_bias),
                            nn.Tanh()]
        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, x):
        # Encode
        x = self.encoder(x)

        # Transform
        x = self.residuals(x)

        # Decode
        x = self.decoder(x)
        return x

"""
# 64 - 128 - 256
Z = random.random((1, 3, 256, 256))
G = Generator(3, 3)
Z = torch.from_numpy(Z).float()
G(Z)
"""

