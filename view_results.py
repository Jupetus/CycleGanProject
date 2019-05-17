import numpy as np

import matplotlib.pyplot as plt

from models.generator import Generator

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

G_Y = Generator(3, 3, nn.InstanceNorm2d)
G_Y.load_state_dict(torch.load("./outputs/models/G_Y", map_location='cpu'))
G_Y.to(device)

# Data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

landscape_folder = torchvision.datasets.ImageFolder("./../data/landscapes/", transform)
landscape_loader = torch.utils.data.DataLoader(landscape_folder, batch_size=batch_size, shuffle=True)


test = iter(landscape_loader)
img_data,_ = next(test) 
original1 = np.array(img_data[0])
fake1 = np.array(G_Y(img_data[0].view(1, 3, 128, 128).to(device)).cpu().detach())[0]

original2 = np.array(img_data[1])
fake2 = np.array(G_Y(img_data[1].view(1, 3, 128, 128).to(device)).cpu().detach())[0]

original3 = np.array(img_data[2])
fake3 = np.array(G_Y(img_data[2].view(1, 3, 128, 128).to(device)).cpu().detach())[0]

# Transform to nice horsy  !!!!!! better transpose: transpose(1,2,0) !!!!!!
original1 = original1.transpose()
original1 = original1.transpose((1,0,2))
original1 = original1 / 2 + 0.5

fake1 = fake1.transpose()
fake1 = fake1.transpose((1,0,2))
fake1 = fake1 / 2 + 0.5

original2 = original2.transpose()
original2 = original2.transpose((1,0,2))
original2 = original2 / 2 + 0.5

fake2 = fake2.transpose()
fake2 = fake2.transpose((1,0,2))
fake2 = fake2 / 2 + 0.5

original3 = original3.transpose()
original3 = original3.transpose((1,0,2))
original3 = original3 / 2 + 0.5

fake3 = fake3.transpose()
fake3 = fake3.transpose((1,0,2))
fake3 = fake3 / 2 + 0.5

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
ax1.imshow(original1)
ax2.imshow(fake1)

ax3.imshow(original2)
ax4.imshow(fake2)

ax5.imshow(original3)
ax6.imshow(fake3)

plt.show()
