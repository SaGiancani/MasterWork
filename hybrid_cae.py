import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import distutils.dir_util

class HybridConvolutionalAutoencoder(nn.Module):      
    def __init__(self):
        # Input Dataset built with images 128x128 
        super(HybridConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=0, stride=1), #127x127
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(3, 3),            
            
            nn.Conv2d(4, 16, 3, padding=0, stride=1), #31x31
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, padding=0, stride=1), #30x30
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 3),            
            
            nn.Conv2d(32, 64, 3, padding=0, stride=1), #7x7
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, padding=0, stride=1), #6x6
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, 3),            
            
        )
        
        self.linear_e = nn.Sequential(
            nn.Linear(2*2*128, 16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.ReLU(),
        )
        
        self.linear_d = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2*2*128),
        )
        
        self.decoder = nn.Sequential(                                  
            nn.ConvTranspose2d(128, 64, 4,  stride=2),   #7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(64, 32, 4, stride=2),   #15x15
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(32, 16, 4, stride=2),   #31x31
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 4, 4, stride=2),  #63x63
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),  
            
            nn.ConvTranspose2d(4, 4, 4, stride=2),  #63x63
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),            
            
            nn.ConvTranspose2d(4, 1, 3, stride=1),  #63x63
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
        
        
#Understanding the difficulty of training deep feedforward neural networks - Glorot, Xavier & Bengio, Y. (2010)
def xavier_init_uniform(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

def xavier_init_normal_(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()

#Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)        
def kaiming_init_normal_(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        m.bias.data.zero_()

def kaiming_init_uniform(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        m.bias.data.zero_()