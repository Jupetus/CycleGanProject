from models.discriminator import Discriminator
from models.generator import Generator
from train import Trainer

import torch

import argparse
import os

# Convert command line str argument to bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

# Read traning config and run!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataX', type=str, default="./../data/horse2zebra/horses")
    parser.add_argument('--dataY', type=str, default="./../data/horse2zebra/zeepra")
    parser.add_argument('--loadModels', type=str2bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=1)
    
    parser.add_argument('--cycleLoss', type=str2bool, default=True)
    parser.add_argument('--cycleMultiplier', type=int, default=10)

    parser.add_argument('--identityLoss', type=str2bool, default=True)
    parser.add_argument('--identityMultiplier', type=int, default=5)
    
    config = parser.parse_args()
    
    trainer = Trainer(config)
    trainer.train()