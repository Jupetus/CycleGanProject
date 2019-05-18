import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random

class Discriminator(nn.Module):
    def __init__(self, input_dimension):
        super(Discriminator, self).__init__()

        # Fixed parameters
        hidden_dimension = 64
        norm_layer = nn.BatchNorm2d
        use_bias = norm_layer == nn.InstanceNorm2d

        self.convolutions = nn.Sequential(
          nn.Conv2d(input_dimension, hidden_dimension, 4, 2, 1, bias=use_bias),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Conv2d(hidden_dimension, 2*hidden_dimension, 4, 2, 1, bias=use_bias),
          norm_layer(2*hidden_dimension),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Conv2d(2*hidden_dimension, 4*hidden_dimension, 4, 2, 1, bias=use_bias),
          norm_layer(4*hidden_dimension),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Conv2d(4*hidden_dimension, 8*hidden_dimension, 4, 2, 1, bias=use_bias),
          norm_layer(8*hidden_dimension),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Conv2d(8*hidden_dimension, 8*hidden_dimension, 4, 1, 1, bias=use_bias),
          norm_layer(8*hidden_dimension),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Conv2d(8*hidden_dimension, 1, 4, 1, 1),
        )


    def forward(self, x):
        x = self.convolutions(x)
        return x