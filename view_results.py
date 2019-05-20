import numpy as np

import matplotlib.pyplot as plt
from models.generator import Generator

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

import argparse
import os

"""
Transform random image(s) from the input folder with generator thats given as input
"""


# Reads traning config and run!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="./../data/horse2zebra/horses")
    parser.add_argument('--model', type=str, default="./outputs/models/G_Y")
    parser.add_argument('--sample_n_images', type=int, default=1)
    
    image_width = image_height = 256
    config = parser.parse_args()
    output_folder = "./outputs/results/"

    batch_size = config.sample_n_images
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = Generator(3, 3, nn.InstanceNorm2d) # You might have to change norm by yourself!
    G.load_state_dict(torch.load(config.model, map_location='cpu'))
    G.to(device)

    # Data
    transform = transforms.Compose([
        transforms.Resize((image_width, image_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_folder = torchvision.datasets.ImageFolder("./../data/horse2zebra/horses", transform)
    data_loader = torch.utils.data.DataLoader(data_folder, batch_size=batch_size, shuffle=True)

    counter = 0
    image_iter = iter(data_loader)
    first_batch,_ = next(image_iter)
    for image in first_batch:
        original = np.array(image)
        generated = np.array(G(image.view(1, 3, image_width, image_height).to(device)).cpu().detach())[0]

        # Transform images back to (0-1) range
        original = original.transpose((1,2,0))
        original = original / 2 + 0.5
        generated = generated.transpose((1,2,0))
        generated = generated / 2 + 0.5

        plt.imsave('{}/{}_original_image.png'.format(output_folder, counter), original)
        plt.imsave('{}/{}_generated_image.png'.format(output_folder, counter), generated)

        counter += 1
