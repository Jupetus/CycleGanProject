import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random

class residual_block(nn.Module):
    def __init__(self, hidden_dimension, use_bias, norm):
        super(residual_block, self).__init__()
    
        # Residual (Transform)
        self.residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4*hidden_dimension, 4*hidden_dimension, 3 ,1, 0, bias=use_bias),
            norm(4*hidden_dimension),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4*hidden_dimension, 4*hidden_dimension, 3 ,1, 0, bias=use_bias),
            norm(4*hidden_dimension),
        )

    def forward(self, x):
        residual_relu = False
        if residual_relu:
            return F.relu(x + self.residual(x))
        else:
            return x + self.residual(x)


# Transform image between classes: X -> Y
class Generator(nn.Module):
    def __init__(self, input_dimension, output_dimension, norm=nn.BatchNorm2d, n_residuals=6):
        super(Generator, self).__init__()


        use_bias = norm == nn.InstanceNorm2d
        hidden_dimension = 64
        
        # Make Encoder
        # Layer1
        self.encoder = nn.Sequential(
          nn.ReflectionPad2d(3),
          nn.Conv2d(input_dimension, hidden_dimension, 7, 1, 0, bias=use_bias),
          norm(hidden_dimension),
          nn.ReLU(inplace=True),

          nn.ReflectionPad2d(1),
          nn.Conv2d(hidden_dimension, 2*hidden_dimension, 3, 2, 0, bias=use_bias),
          norm(2*hidden_dimension),
          nn.ReLU(inplace=True),

          nn.ReflectionPad2d(1),
          nn.Conv2d(2*hidden_dimension, 4*hidden_dimension, 3, 2, 0, bias=use_bias),
          norm(4*hidden_dimension),
          nn.ReLU(inplace=True),
        )

        # Make Residuals ... transform the image
        res_layers = []
        for _ in range(n_residuals):
            res_layers.append(residual_block(hidden_dimension, use_bias, norm))
        self.residuals = nn.Sequential(*res_layers)

        # Make Decoder
        # Decode
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(4*hidden_dimension, 2*hidden_dimension, 3, 2, 1, output_padding=1, bias=use_bias),
            norm(2*hidden_dimension),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2*hidden_dimension, hidden_dimension, 3, 2, 1, output_padding=1, bias=use_bias),
            norm(hidden_dimension),
            nn.ReLU(inplace=True),
		
            nn.ReflectionPad2d(3),
            nn.Conv2d(hidden_dimension, output_dimension, 7, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)

        # Transform
        x = self.residuals(x)

        # Decode
        x = self.decoder(x)
        return x


