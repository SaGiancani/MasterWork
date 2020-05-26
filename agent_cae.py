import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from itertools import count

import hybrid_cae 
import hybrid_32_cae
import hybrid_64_cae
import pure_cae 
import utilities as u
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
        self.mode_cae = self.h['mode']
        self.val_share = self.h['percentage_of_dataset_for_eval']
        self.interval = self.h['interval_for_print']
        self.sched_mode = self.h['lr_scheduler_mode']
        
        #Path of the dataset
        self.path_dataset = self.h['path_dataset']
        
        #Number of steps per evaluated image
        self.num_steps = self.h['steps_for_each_image']
        
        #Initialization dataset
        self.images_set, self.eval_set, self.img_size = load_data.init_dataset_planar(self.batch_size,
                                                                                      self.val_share,
                                                                                      dataset_file_path = self.path_dataset)
        
        #Utility lists
        #Train
        self.img_array = []
        self.epoch_losses = []
        #Evaluate
        self.evaluated_losses = []
        
        # Loading architecture 
        if self.mode_cae[0] == 'pure':
            #CAE
            self.model = pure_cae.ConvolutionalAutoencoder().to(self.device)
        if (self.mode_cae[0] == 'hybrid') and (self.mode_cae[1] == 128):
            #Hybrid
            self.model = hybrid_cae.HybridConvolutionalAutoencoder().to(self.device)
        if (self.mode_cae[0] == 'hybrid') and (self.mode_cae[1] == 64):
            self.model = hybrid_64_cae.HybridConvolutionalAutoencoder().to(self.device)
        if (self.mode_cae[0] == 'hybrid') and (self.mode_cae[1] == 32):
            self.model = hybrid_32_cae.HybridConvolutionalAutoencoder().to(self.device)           
       
        #Initialization method
        if ('initialization' in self.h.keys()) and (self.h['initialization'] == 'xavier_normal'):
            self.model.apply(u.xavier_init_normal_)
            
        if ('initialization' in self.h.keys()) and (self.h['initialization'] == 'xavier_uniform'):
            self.model.apply(u.xavier_init_uniform)
        
        #Initialization optimizer and loss-computation algorithm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        # Cyclical Learning Rates for Training Neural Networks, Leslie N. Smith
        # or simple exponential decay, gamma-based
        self.lr_list = []
        if self.sched_mode:
            self.gamma = self.h['gamma']
            if ('base_learning_rate' in self.h.keys()) and ('max_learning_rate' in self.h.keys()):
                self.lr_base = self.h['base_learning_rate']
                self.lr_max = self.h['max_learning_rate']
                self.scheduler_lr = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                                      self.lr_base,
                                                                      self.lr_max,
                                                                      step_size_up=self.epochs/10,
                                                                      step_size_down=None,
                                                                      cycle_momentum = False,
                                                                      mode='exp_range',
                                                                      gamma=self.gamma)
            else:
                self.scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=self.gamma) 
        else:
            # Appending the lr for printing
            self.lr_list.append(self.lr)
        
        
    def images_during_training(self, output, epoch, index_img = 3):
        #Printing images during training: the images are not ready yet
        if (epoch%(self.epochs/self.num_steps) == 0):
            # use detach when it's an output that requires_grad
            tmp = output.cpu().detach().numpy()
            self.img_array.append(tmp[index_img])
            
    def build_batch(self):
        # obtain one batch of test images
        dataiter = iter(self.images_set)
        images_for_test_, _ = dataiter.next()
        images_for_test = images_for_test_.unsqueeze(1)
        print('images_for_test shape:' + str(images_for_test.shape))
        return images_for_test

    # Training function
    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs+1):
        #for epoch in trange(1, self.epochs+1):
            for i, data in enumerate(self.images_set):
                self.optimizer.zero_grad()
                img_ = data[0].to(self.device)
                img = img_.unsqueeze(1)
                # We don't utilize the target data[1]
                output = self.model(img)
                loss_ = self.loss(output, img)
                loss_.backward()
                self.optimizer.step()
            if self.sched_mode:
                self.scheduler_lr.step()
                #print(self.optimizer.param_groups[0]['lr']) 
                self.lr_list.append(self.optimizer.param_groups[0]['lr'])
            #Printing not yet trained images, one every num_steps 
            loss_value = loss_.item()
            if (epoch%self.interval == 0) and (epoch>0) and ((self.mode_cae[1] == 64) or (self.mode_cae[1] == 32)):
                print('Train Epoch: {}, Loss: {:.6f}, Learning Rate: {:.6f}'.format(epoch, loss_, self.lr_list[-1]))
            if (epoch%self.interval == 0) and (epoch>0) and (self.mode_cae[1] != 64) and (self.mode_cae[1] != 32):
                print('Train Epoch: {}, Loss: {:.2f}, Learning Rate: {:.6f}'.format(epoch, loss_, self.lr_list[-1]))           
            self.images_during_training(output, epoch)
            self.epoch_losses.append(loss_value)
            # Evaluate loss
            self.eval_loss()
            if (epoch >= self.epochs):
                self.model.save(self.h['name'])
            
    def eval_loss(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.eval_set):
                img_ = data[0].to(self.device)
                img = img_.unsqueeze(1)
                # We don't utilize the target data[1]
                output = self.model(img)
                loss_ = self.loss(output, img)
            loss_value = loss_.item()
            self.evaluated_losses.append(loss_value)
            if (len(self.evaluated_losses)%self.interval == 0) and (self.mode_cae[1] != 64) and (self.mode_cae[1] != 32):
                avg_score = np.mean(self.evaluated_losses[max(0, len(self.evaluated_losses)-self.interval):\
                                                         (len(self.evaluated_losses)+1)])
                print('Valutation mean loss: {:.2f}'.format(avg_score))
            if (len(self.evaluated_losses)%self.interval == 0) and ((self.mode_cae[1] == 64) or (self.mode_cae[1] == 32)):
                avg_score = np.mean(self.evaluated_losses[max(0, len(self.evaluated_losses)-self.interval):\
                                                         (len(self.evaluated_losses)+1)])
                print('Valutation mean loss: {:.6f}'.format(avg_score))
            self.model.train()
            return

    def evaluation(self, mode, imgs):
        #Set eval mode on
        self.model.eval()
        with torch.no_grad():
            if type(imgs) is np.ndarray:
                imgs = torch.Tensor(imgs)
            
            if (mode == 'model'):
                # get sample outputs for the pure convolutional model
                output = self.model(imgs.to(self.device))
                
            if (mode == 'encoder'):
                # get sample outputs for the encoder of the pure convolutional model
                output = self.model.encoder(imgs.to(self.device))
                #self.encoder_out = output_pure
                if (self.mode_cae[0] == 'hybrid'):
                    if (self.mode_cae[1] != 32):
                        tmp = output.view(-1, 2*2*256)
                    if (self.mode_cae[1] == 32):
                        tmp = output.view(-1, 2*2*128)
                    output = self.model.linear_e(tmp.to(self.device))
                
            if (mode == 'decoder'):
                if (self.mode_cae[0] == 'hybrid'):
                    tmp = self.model.linear_d(imgs.to(self.device))
                    if (self.mode_cae[1] != 32):            
                        imgs = tmp.view(-1, 256, 2, 2)
                    if (self.mode_cae[1] == 32):
                        imgs = tmp.view(-1, 128, 2, 2)
                # get sample outputs for the decoder of the pure convolutional model
                output = self.model.decoder(imgs.to(self.device))
                
            if (mode != 'model') and (mode != 'encoder') and (mode != 'decoder'):
                print('Warning: The mode choosen is wrong')
    
            # prep images for display
            # use detach when it's an output that requires_grad
            images_for_display = imgs.detach().cpu().numpy()
            output_for_display = output.cpu().detach().numpy()
            
            return output_for_display, images_for_display