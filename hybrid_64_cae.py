import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import distutils.dir_util

class HybridConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(HybridConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=0, stride=1), #127x127
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 3),            
            
            nn.Conv2d(32, 64, 3, padding=0, stride=1), #31x31
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, padding=0, stride=1), #30x30
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 3),            
            
            nn.Conv2d(128, 128, 2, padding=0, stride=1), #7x7
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 2, padding=0, stride=1), #6x6
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, 2, padding=0, stride=1), #6x6
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(256),
        )
        
        self.linear_e = nn.Sequential(
            nn.Linear(2*2*256, 16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.ReLU(),
        )
        
        self.linear_d = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2*2*256),
        )
    
        self.decoder = nn.Sequential(                                  
            nn.ConvTranspose2d(256, 256, 4,  stride=2),   #7x7
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(256, 128, 4, stride=2),   #15x15
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(128, 128, 4, stride=2),   #31x31
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2),  #63x63
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  
            
            nn.ConvTranspose2d(64, 1, 3, stride=1),  #63x63
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.encoder(x)
        #print('encoder(x): '+str(x.shape))
        x = x.view(-1, 2*2*256)
        x = self.linear_e(x)
        #print('linear e: '+str(x.shape))
        x = self.linear_d(x)
        #print('linear d: '+str(x.shape))
        x = x.view(-1, 256, 2, 2)
        x = self.decoder(x)
        #print('decoder(x): '+str(x.shape))
        #x = F.sigmoid(x)
        return x

    def save(self, filename):
        # Save the model state
        distutils.dir_util.mkpath(filename)
        torch.save(self.state_dict(), filename+'/'+filename +'_model.pt')
        