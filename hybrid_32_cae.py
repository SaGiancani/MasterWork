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
            
            nn.Conv2d(64, 128, 2, padding=0, stride=1), #30x30
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 128, 2, padding=0, stride=1), #30x30
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(128)
        )
        
        self.linear_e = nn.Sequential(
            nn.Linear(2*2*128, 10),
            nn.ReLU(),
            #nn.Linear(16,2),
            #nn.ReLU(),
        )
        
        self.linear_d = nn.Sequential(
            #nn.Linear(2, 16),
            #nn.ReLU(inplace=True),
            nn.Linear(10, 2*2*128),
        )
        
        self.decoder = nn.Sequential(                                  
            nn.ConvTranspose2d(128, 128,2,  stride=1),   #7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
           
            nn.ConvTranspose2d(128, 64, 2, stride=2),   #15x15
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(64, 64, 2, stride=1),   #31x31
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=1),  #63x63
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 32, 4, stride=3),   #31x31
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, 5, stride=1),   #31x31
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):      
        x = self.encoder(x)
        #print('encoder(x): '+str(x.shape))
        x = x.view(-1, 2*2*128)
        x = self.linear_e(x)
        #print('linear e: '+str(x.shape))
        x = self.linear_d(x)
        #print('linear d: '+str(x.shape))
        x = x.view(-1, 128, 2, 2)
        x = self.decoder(x)
        #print('decoder(x): '+str(x.shape))
        #x = F.sigmoid(x)
        return x
    
    def save(self, filename):
        # Save the model state
        distutils.dir_util.mkpath(filename)
        torch.save(self.state_dict(), filename+'/'+filename +'_model.pt')
        