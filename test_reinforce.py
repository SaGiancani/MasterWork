import gym
import gym_control_model.planararm.register_env
import reinforce_algorithm as ra
import plot as plot
import utilities as util

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from itertools import count
import torch

import time
import sys


# Print the starting time
print(time.asctime(time.localtime(time.time())))    
    
# Define a seed for reproducability and the name of the model, hyper parameters files
filename = str(time.asctime(time.localtime(time.time())))
environment = 'CartPole-v0'
#environment = 'PlanarArmTeacher2Learner3-v1'
my_seed = 1234
n_joints = (2,3)

# Choose the select_action algorithm: you can choose between 'regular' and 'greed'
select_action_algorithm = 'greed'

# Choose the mode: with or without Baseline approach
baseline = False
#baseline = True

# Initialization of the max_episodes, max_steps variables:
# Max Steps threshold set to 15000
max_episodes, max_steps, interval = 3000, 300, 100

# The interval of train episodes between an evaluation and other
evaluation_interval = 5

# Initializiation of the max_episodes_eval, max_steps_eval variables:
max_episodes_eval, max_steps_eval = 10, max_steps

# Initialize an epsilon value for the greed algorithm, the learning rate, gamma
epsilon = 0.1
# With baseline fits better 1e-4
if baseline:
    lr = 1e-3
# With regular REINFORCE is better 1e-3
else:
    lr = 1e-3
gamma = 0.9

# A list for plotting, storing the hyperparameters and the corresponding results
results_neural_network = list()
results_neural_network_meanstd = list()
storing_list = list()

# Initialize TensorBoard for the visualization of the data
tboard = SummaryWriter()

# Start a time counter
start_time = time.time()

# Set the seed in torch and numpy
torch.manual_seed(my_seed)
np.random.seed(my_seed)
    
#Initializing the gym environment

env = gym.make(environment)

# and set this seed in gym
env.seed(my_seed)

actions = env.action_space.n
if environment=='PlanarArmTeacher2Learner3-v1':
    states = 10
if environment=='CartPole-v0':
    states = 4
        
#Instantiate the hyperparameters
hyperparam_dict = {'name': filename, 'n_joints':n_joints, 
                   'actions': actions,'states': states, 'gamma':gamma, 
                   'learning_rate':lr, 'max_steps':max_steps, 'max_episodes':max_episodes, 
                   'my_seed':my_seed, 'select_action_algorithm':select_action_algorithm,
                   'max_episodes_eval':max_episodes_eval, 'max_steps_eval': max_steps_eval, 
                   'evaluation_interval':evaluation_interval}

#In the case in which the select action algorithm was greed, I add an epsilon parameter to the dict
if select_action_algorithm == 'greed':
    hyperparam_dict['epsilon'] = epsilon

# If the baseline mode is on, two more hyperparameters have to be implemented: the boolean switch and the length number
hyperparam_dict['baseline'] = baseline
if baseline:
    #hyperparam_dict['baseline_dimension'] = baseline_length
    title = 'REINFORCE with baseline'
else:
    title = 'REINFORCE'

# Initialize the Agent
agent = ra.REINFORCE_Agent(hyperparam_dict)

# Execute the REINFORCE algorithm 
ep_rewards, running_rewards, mean_rew, std_rew = ra.reinforce(max_steps, max_episodes,
                                                              env, agent, render=False,
                                                              mode='train', chooser=None,
                                                              log_interval=interval, tb=tboard)

# Store the results
#results_neural_network.append((ep_rewards, 'Episode Rewards', 0, 'dash'))
#results_neural_network.append((running_rewards, 'Running Rewards', 0, 'regular'))
results_neural_network.append((running_rewards, 'Running Reward', 4, 'regular', ep_rewards))
results_neural_network_meanstd.append((mean_rew, 'Std/Mean Reward Evaluation', 2, 'regular',std_rew ))
if baseline:
    results_neural_network.append((agent.baseline, 'Baseline', 1, 'regular'))
    results_neural_network_meanst.append((agent.baseline, 'Baseline', 1, 'regular'))
# Appending for the handling of the figure number
storing_list.append(results_neural_network)
storing_list.append(results_neural_network_meanstd)
# Plot cumulative rewards, standard deviation and evaluated average episodic reward.
for k, i in enumerate(storing_list):
    plot.multi_plot(i, hyperparam_dict, k, title)

# Saving the hyperparam_dictionary into a file
hyperparam_dict['total_time']= (time.time() - start_time)/60
util.save_dict_txt(hyperparam_dict)

# Calculate the total time of execution
print('The total time to compile '+str(max_episodes)+' episodes for a maximum of '+ str(max_steps) + ' steps is: ' + str(hyperparam_dict['total_time']))

