import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random

class Discriminator(nn.Module):
    def __init__(self, input_dimension):
        super(Discriminator, self).__init__()

        hidden_dimension = 64

        self.convolutions = nn.Sequential(
          nn.Conv2d(input_dimension, hidden_dimension, 4, 2, 0, bias=True),
          #nn.BatchNorm2d(hidden_dimension),
          nn.ReLU(inplace=False),

          nn.Conv2d(hidden_dimension, hidden_dimension, 4, 2, 0, bias=True),
          nn.BatchNorm2d(hidden_dimension),
          nn.ReLU(inplace=False),

          nn.Conv2d(hidden_dimension, 2*hidden_dimension, 4, 2, 0, bias=True),
          nn.BatchNorm2d(2*hidden_dimension),
          nn.ReLU(inplace=False),

          nn.Conv2d(2*hidden_dimension, 2*hidden_dimension, 4, 2, 0, bias=True),
          nn.BatchNorm2d(2*hidden_dimension),
          nn.ReLU(inplace=False),
        )

        self.output = nn.Sequential(
          nn.Linear(4608, 512),
          nn.ReLU(),
          nn.Linear(512, 32),
          nn.ReLU(),
          nn.Linear(32, 1),
          nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 4608)
        x = self.output(x)
        return x

"""
Z = random.random((1, 3, 128, 128))
D = Discriminator(3)
Z = torch.from_numpy(Z).float()
D(Z) 
"""