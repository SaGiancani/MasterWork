import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from itertools import count

import pure_cae 
import load_data
import time

import os
from tqdm import trange

class CAE_Agent(object):
    def __init__(self, h):
        #Hyperparameters
        self.h = h
        self.batch_size = self.h['batch_size']
        self.lr = self.h['learning_rate']
        self.epochs = self.h['max_episodes']
        self.test_batch_num = self.h['test_batch_num']
        self.device = self.h['device']
        
        #Path of the dataset
        self.path_dataset = self.h['path_dataset']
        #self.path_dataset = "learner_teacher_10k.npz"
        
        #Number of steps per evaluated image
        self.num_steps = self.h['steps_for_each_image']
        #self.num_steps = 10
        
        #Initialization dataset
        self.images_set, self.img_size = load_data.init_dataset_planar(self.batch_size,
                                                                       dataset_file_path = self.path_dataset)
        
        #Utility lists
        #Train
        self.img_array = []
        self.epoch_losses = []

        #Reconstruction
        # Array to store the original test images
        self.testdata_input = np.zeros((len(self.images_set), self.batch_size, 1, self.img_size, self.img_size))
        # Array to store the original test images
        self.testdata_output = np.zeros((1, self.img_size, self.img_size))
        # Array to store the code 2x1x1:
        self.encoder_out = torch.zeros((self.batch_size, 2, 1, 1))
        
        #CAE
        self.model_pure = pure_cae.ConvolutionalAutoencoder().to(self.device)
        
        #Initialization method
        if ('initialization' in self.h.keys()) and (self.h['initialization'] == 'xavier_normal'):
            self.model_pure.apply(pure_cae.xavier_init_normal_)
            
        if ('initialization' in self.h.keys()) and (self.h['initialization'] == 'xavier_uniform'):
            self.model_pure.apply(pure_cae.xavier_init_uniform)
        
        #Initialization optimizer and loss-computation algorithm
        self.optimizer_pure = torch.optim.Adam(self.model_pure.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
        
    def images_during_training(self, output, epoch, index_img = 3):
        #Printing images during training: the images are not ready yet
        if (epoch%(self.epochs/self.num_steps) == 0):
            # use detach when it's an output that requires_grad
            tmp = output.cpu().detach().numpy()
            self.img_array.append(tmp[index_img])
            
    # Reconstruction function
    def reconstruction_encode(self, epoch, i, img):
        self.testdata_input[i] = img.detach().cpu().numpy()
        # latent from pure convolutional encoder and storing
        latent_pure = self.model_pure.encoder(img)
        self.testdata_latent_pure[i] = latent_pure
        
        # Encode the images of the test set using the encoder trained on the train set
        for i,test_data in enumerate(test_image_dataloader):
            # Preparation original images
            data = test_data[0].to(device)
            #print(data.shape)
            img = data.unsqueeze(1)
            #print(img.shape)
        
            testdata_input[i] = img.detach().cpu().numpy()
            
            # latent from pure convolutional encoder and storing
            latent_pure = model_pure.encoder(img)
            testdata_latent_pure[i] = latent_pure
            #if i==4:
            #    break
        
            
    def reconstruction_decode(self, latents_pure):
        # Set the autoencoder networks to evaluation mode
        self.model_pure.eval()

        # Display decoded image from convolutional Autoencoder
        img_pure = self.model_pure.decoder(latents_pure.to(self.device))
        self.testdata_output[i] = img_pure[i].cpu().detach().numpy().reshape(128,128)
        
    def build_batch(self):
        # obtain one batch of test images
        dataiter = iter(self.images_set)
        images_for_test_, _ = dataiter.next()
        images_for_test = images_for_test_.unsqueeze(1)
        print('images_for_test shape:' + str(images_for_test.shape))
        return images_for_test

    # Training function
    def train(self):
        for epoch in trange(1, self.epochs+1):
            for i, data in enumerate(self.images_set):
                img_ = data[0].to(self.device)
                img = img_.unsqueeze(1)
                # We don't utilize the target data[1]
                output = self.model_pure(img)
                loss_ = self.loss(output, img)
                self.optimizer_pure.zero_grad()
                loss_.backward()
                self.optimizer_pure.step()
                #Reconstruction loop inside the train loop: saving gpu store capacity 
                # This kind of strategy is the worst for the gpu storage
                #if epoch==self.epochs:
                    #self.reconstruction_encode(epoch, i, img)
            if epoch >= self.epochs:
                self.model_pure.save(self.h['name'])
            #Printing not yet trained images, one every num_steps 
            self.images_during_training(output, epoch)
            loss_value = loss_.item()
            self.epoch_losses.append(loss_value)
            
    def evaluation(self, mode, imgs):
        #Set eval mode on
        self.model_pure.eval()

        if type(imgs) is np.ndarray:
            imgs = torch.Tensor(imgs)
            
        if (mode == 'model'):
            # get sample outputs for the pure convolutional model
            output_pure = self.model_pure(imgs.to(self.device))
            
        if (mode == 'encoder'):
            # get sample outputs for the encoder of the pure convolutional model
            output_pure = self.model_pure.encoder(imgs.to(self.device))
            #self.encoder_out = output_pure            
            
        if (mode == 'decoder'):
            # get sample outputs for the decoder of the pure convolutional model
            output_pure = self.model_pure.decoder(imgs.to(self.device))
            
        if (mode != 'model') and (mode != 'encoder') and (mode != 'decoder'):
            print('Warning: The mode choosen is wrong')

        # prep images for display
        # use detach when it's an output that requires_grad
        images_for_display = imgs.detach().cpu().numpy()
        output_for_display = output_pure.cpu().detach().numpy()
        
        return output_for_display, images_for_display
