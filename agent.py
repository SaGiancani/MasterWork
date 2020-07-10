import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from itertools import count

import hybrid_32_cae
import utilities as u
import q_learning as qlt
import reinforce_algorithm as ra
import dataset.gen_data as gen
import time

import os
from tqdm import trange

class Agent(object):
    def __init__(self,hyperparam_dict):
        #Load trained AutoEncoder
        self.h = hyperparam_dict
        #Images hyperparameters
        self.img_size = self.h['img_size']
        self.img_binary = self.h['img_binary']
        self.dtype_num = self.h['type']
        self.kind_alg = self.h['algorithm'] 
    
        #Loading the modules
        self.model_cae = hybrid_32_cae.HybridConvolutionalAutoencoder.load('32bitSetModel.pt')
        # Choose the algorithm
        if self.kind_alg == 'DQN':
            self.agent = qlt.DQN_Agent(hyperparam_dict)
        if self.kind_alg == 'REINFORCE':
            self.agent = ra.REINFORCE_Agent(hyperparam_dict)

    def dimension_reduction(self, image):
        return self.model_cae.forward_bottleneck(image)
    
    def join_observations(self, tup):
        tmp = torch.cat((tup[0], tup[1]), 0)
        tmp = tmp.view(1, -1)
        return tmp.detach().numpy()

    def vision_module(self, env, mode, action):
        if mode == 'reset':
            observation = env.reset()
            observation = gen.generate_images_agent(observation, self.img_size, self.img_binary, self.dtype_num)
            tmp_o = torch.Tensor(observation[0]).unsqueeze(0).unsqueeze(0)
            tmp_i = torch.Tensor(observation[1]).unsqueeze(0).unsqueeze(0)
            learner = self.dimension_reduction(tmp_o)
            teacher = self.dimension_reduction(tmp_i)
            return (learner, teacher)
            
        if mode == 'step':
            observation, reward, done, _ = env.step(action)
            observation = gen.generate_images_agent(observation, self.img_size, self.img_binary, self.dtype_num)
            tmp_o = torch.Tensor(observation[0]).unsqueeze(0).unsqueeze(0)
            tmp_i = torch.Tensor(observation[1]).unsqueeze(0).unsqueeze(0)
            learner = self.dimension_reduction(tmp_o)
            teacher = self.dimension_reduction(tmp_i)
            return (learner, teacher), reward, done