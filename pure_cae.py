import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import distutils.dir_util

class ConvolutionalAutoencoder(nn.Module):
        
    def __init__(self):
        # Input Dataset built with images 128x128 
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0, stride=1), #127x127
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),            
            
            nn.Conv2d(16, 16, 3, padding=0, stride=1), #31x31
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, 3, padding=0, stride=1), #30x30
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),            
            
            nn.Conv2d(16, 4, 3, padding=0, stride=1), #7x7
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(4),
            
            nn.Conv2d(4, 4, 3, padding=0, stride=1), #6x6
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2, 2),            
            
            nn.Conv2d(4, 4, 3, padding=0, stride=1), # 2x2
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(4),
            
            nn.Conv2d(4, 2, 3, padding=0, stride=1), # 1x1
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(2),
            nn.MaxPool2d(2, 2),  
            
            nn.Conv2d(2, 2, 3, padding=0, stride=1), # 1x1
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(2),
            
            nn.Conv2d(2, 2, 2, padding=0, stride=1), # 1x1
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(2),
               
        )
        
        self.decoder = nn.Sequential(                       
            nn.ConvTranspose2d(2, 2, 3, stride=1),   #3x3
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2, 2, 2, stride=1),   #3x3
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2, 4, 2,  stride=1),   #7x7
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(4, 4, 3, stride=1),   #7x7
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(4, 4, 3, stride=2),   #15x15
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(4, 16, 3, stride=2),   #31x31
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 16, 3, stride=2),  #63x63
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 16, 3, stride=2), #127x127
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 1, 2, stride=1),  #128x128
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        #x = F.sigmoid(x)
        return x

    def save(self, filename):
        # Save the model state
        distutils.dir_util.mkpath(filename)
        torch.save(self.state_dict(), filename+'/'+filename +'_model.pt')
        