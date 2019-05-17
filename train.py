from models.discriminator import Discriminator
from models.generator import Generator

from torchvision import transforms
import torch.nn as nn
import torchvision
import torch

import numpy as np

# define the initialization function for network weights
def init_func(m):  
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1: 
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Trainer():
    def __init__(self, config):
        self.batch_size = config.batchSize
        self.epochs = config.epochs
        self.use_cycle_loss = config.cycleLoss
        self.use_identity_loss = config.identityLoss
        self.load_models = config.loadModels
        self.data_x_loc = config.dataX
        self.data_y_loc = config.dataY

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        self.model_path = "./outputs/"

        self.init_models()
        self.init_data_loaders()
        self.g_optimizer = torch.optim.Adam(list(self.G_X.parameters()) + list(self.G_Y.parameters()), lr=config.lr)
        self.d_optimizer = torch.optim.Adam(list(self.D_X.parameters()) + list(self.D_Y.parameters()), lr=config.lr)
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=1, gamma=0.95)


    # Load/Construct the models
    def init_models(self):
        
        self.G_X = Generator(3, 3, nn.InstanceNorm2d)
        self.D_X = Discriminator(3)
        self.G_Y = Generator(3, 3, nn.InstanceNorm2d)
        self.D_Y = Discriminator(3)
        
        if self.load_models:
            self.G_X.load_state_dict(torch.load(self.model_path + "models/G_X", map_location='cpu'))
            self.G_Y.load_state_dict(torch.load(self.model_path + "models/G_Y", map_location='cpu'))
            self.D_X.load_state_dict(torch.load(self.model_path + "models/D_X", map_location='cpu'))
            self.D_Y.load_state_dict(torch.load(self.model_path + "models/D_Y", map_location='cpu'))
        else:
            self.G_X.apply(init_func)
            self.G_Y.apply(init_func)
            self.D_X.apply(init_func)
            self.D_Y.apply(init_func)
            
        self.G_X.to(self.device)
        self.G_Y.to(self.device)
        self.D_X.to(self.device)
        self.D_Y.to(self.device)

    # Get data loaders 
    def init_data_loaders(self):
        img_width = 128
        img_height = 128        

        transform = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        X_folder = torchvision.datasets.ImageFolder(self.data_x_loc, transform)
        self.X_loader = torch.utils.data.DataLoader(X_folder, batch_size=self.batch_size, shuffle=True)

        Y_folder = torchvision.datasets.ImageFolder(self.data_y_loc, transform)
        self.Y_loader = torch.utils.data.DataLoader(Y_folder, batch_size=self.batch_size, shuffle=True)

    # Save the models
    def save_models(self):
        torch.save(self.G_X.state_dict(), self.model_path + "models/G_X")
        torch.save(self.D_X.state_dict(), self.model_path + "models/D_X")
        torch.save(self.G_Y.state_dict(), self.model_path + "models/G_Y")
        torch.save(self.D_Y.state_dict(), self.model_path + "models/D_Y")
    

    # Reset the gradients
    def reset_gradients(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def train(self):
        D_X_losses = []
        D_Y_losses = []

        G_X_losses = []
        G_Y_losses = []

        for epoch in range(self.epochs):
            print("======")
            print("Epoch {}!".format(epoch + 1))
            for (data_X, _), (data_Y, _) in zip(self.X_loader, self.Y_loader):      
                data_X = data_X.to(self.device)
                data_Y = data_Y.to(self.device)

                # Paper reduces lr after 100 epochs
                if epoch > 100:
                    self.scheduler_g.step()
                # =====================================
                # Train Discriminators
                # =====================================
                
                # Train fake X
                self.reset_gradients()
                fake_X = self.G_X(data_Y)
                out_fake_X = self.D_X(fake_X)
                d_x_f_loss = torch.mean(out_fake_X**2) / 2
                d_x_f_loss.backward()
                self.d_optimizer.step()

                # Train fake Y
                self.reset_gradients()
                fake_Y = self.G_Y(data_X)
                out_fake_Y = self.D_Y(fake_Y)
                d_y_f_loss = torch.mean(out_fake_Y**2) / 2
                d_y_f_loss.backward()
                self.d_optimizer.step()

                # Train true X
                self.reset_gradients()
                out_true_X = self.D_X(data_X)
                d_x_t_loss = torch.mean((out_true_X-1) ** 2) / 2
                d_x_t_loss.backward()
                self.d_optimizer.step()

                # Train true Y
                self.reset_gradients()
                out_true_Y = self.D_Y(data_Y)
                d_y_t_loss = torch.mean((out_true_Y-1) ** 2) / 2
                d_y_t_loss.backward()
                self.d_optimizer.step()

                D_X_losses.append([d_x_t_loss.cpu().detach().numpy(), d_x_f_loss.cpu().detach().numpy()])
                D_Y_losses.append([d_y_t_loss.cpu().detach().numpy(), d_y_f_loss.cpu().detach().numpy()])

                # =====================================
                # Train GENERATORS
                # =====================================
                
                # Cycle X -> Y -> X
                self.reset_gradients()

                fake_Y = self.G_Y(data_X)
                out_fake_Y = self.D_Y(fake_Y)

                g_loss1 = torch.mean((out_fake_Y-1)**2)
                if self.use_cycle_loss:
                    reconst_X = self.G_X(fake_Y)
                    g_loss2 = 2 * torch.mean((data_X - reconst_X) ** 2)

                G_Y_losses.append([g_loss1.cpu().detach().numpy(), g_loss2.cpu().detach().numpy()])
                g_loss = g_loss1 + g_loss2
                g_loss.backward()
                self.g_optimizer.step()

                # Cycle Y -> X -> Y
                self.reset_gradients()

                fake_X = self.G_X(data_Y)
                out_fake_X = self.D_X(fake_X)

                g_loss1 = torch.mean((out_fake_X-1)**2)
                if self.use_cycle_loss:
                    reconst_Y = self.G_Y(fake_X)
                    g_loss2 = 2 * torch.mean((data_Y - reconst_Y) ** 2)

                G_X_losses.append([g_loss1.cpu().detach().numpy(), g_loss2.cpu().detach().numpy()])
                g_loss = g_loss1 + g_loss2
                g_loss.backward()
                self.g_optimizer.step()

                # =====================================
                # Train image IDENTITY
                # =====================================

                if self.use_identity_loss:
                    self.reset_gradients()

                    # X should be same after G(X)
                    same_X = self.G_X(data_X)
                    g_loss = torch.mean((data_X - same_X) ** 2)
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Y should be same after G(Y)
                    same_Y = self.G_X(data_Y)
                    g_loss = torch.mean((data_Y - same_Y) ** 2)
                    g_loss.backward()
                    self.g_optimizer.step()

            print("Discriminator: {}".format(D_Y_losses[-5:]))
            print("Generator: {}".format(G_Y_losses[-5:]))
        # Epoch done, save models
        self.save_models()
        np.save(self.model_path + 'losses/G_X_losses.npy', np.array(G_X_losses))
        np.save(self.model_path + 'losses/G_Y_losses.npy', np.array(G_Y_losses))
        np.save(self.model_path + 'losses/D_X_losses.npy', np.array(D_X_losses))
        np.save(self.model_path + 'losses/D_Y_losses.npy', np.array(D_Y_losses))


                
