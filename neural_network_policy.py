import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import distutils.dir_util

class NeuralNetworkPolicy(nn.Module):
    def __init__(self, actions = 27, states = 10):
        super(NeuralNetworkPolicy, self).__init__()
        self.input = states
        self.output = actions
        self.affine1 = nn.Linear(self.input, 256)
        self.dropout1 = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=0.6)
        self.affine3 = nn.Linear(256, self.output)
        # Lists used on REINFORCE Algorithm
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
    # Deleting the droputs raise the performance of the reward function     
        x = self.affine1(x)
        #x = self.dropout1(x)
        x = F.leaky_relu(x)
        x = self.affine2(x)
        #x = self.dropout2(x)
        x = F.leaky_relu(x)
        action_scores = self.affine3(x)
        return action_scores

    def save(self, filename):
        # Save the model state
        distutils.dir_util.mkpath(filename)
        torch.save(self.state_dict(), filename+'/'+filename +'_model.pt')

    @staticmethod
    def load(state_file='policy_network_splittedscripts.pt'):
        # Create a network object with the constructor parameters
        policy = NeuralNetworkPolicy()
        # Load the weights
        policy.load_state_dict(torch.load(state_file))
        # Set the network to evaluation mode
        policy.eval()
        return policy
