import gym
import gym_control_model.planararm.register_env
import agent as a
import q_learning as qlt
import reinforce_algorithm as ra
import plot as plot
import utilities as util

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from itertools import count
import torch

import time
import sys

# Start a time counter
start_time = time.time()

# Print the starting time
print(time.asctime(time.localtime(time.time())))    
    
# Define a seed for reproducability and the name of the model, hyper parameters files
filename = str(time.asctime(time.localtime(time.time())))
environment = 'PlanarArmTeacher2Learner3-v2'
env = gym.make(environment)
my_seed = 1234
n_joints = (2,3)
actions = env.action_space.n
states = 20
lr = 1e-5

img_size = 32
img_binary = True
dtype_num = 32
        
algorithm = 'DQN'
#algorithm = 'REINFORCE'

#Instantiate the hyperparameters
hyperparam_dict = {'name': filename, 
                   'n_joints':n_joints, 
                   'actions': actions,
                   'states': states, 
                   'learning_rate':lr, 
                   'my_seed':my_seed,
                   'environment': environment,
                   'algorithm': algorithm,
                   'type': dtype_num,
                   'img_binary':img_binary,
                   'img_size':img_size}
                   
if algorithm == 'REINFORCE':
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
    gamma = 0.9
    
    # Initialize TensorBoard for the visualization of the data
    tboard = SummaryWriter()
    
    #In the case in which the select action algorithm was greed, I add an epsilon parameter to the dict
    if select_action_algorithm == 'greed':
        hyperparam_dict['epsilon'] = epsilon

    # If the baseline mode is on, two more hyperparameters have to be implemented: the boolean switch and the length number
    hyperparam_dict['max_steps'] = max_steps, 
    hyperparam_dict['max_episodes'] = max_episodes
    hyperparam_dict['baseline'] = baseline
    hyperparam_dict['select_action_algorithm'] = select_action_algorithm
    hyperparam_dict['max_episodes_eval']= max_episodes_eval
    hyperparam_dict['max_steps_eval'] = max_steps_eval
    hyperparam_dict['evaluation_interval'] = evaluation_interval
    hyperparam_dict['gamma'] = gamma 
    
    if baseline:
        #hyperparam_dict['baseline_dimension'] = baseline_length
        title = 'REINFORCE with baseline'
    else:
        title = 'REINFORCE'
        
        
if algorithm == 'DQN':
    gamma = 0.99
    max_episodes, max_steps = 100, 10
    epsilon = 1
    eps_end = 0.01
    eps_dec = 0.996
    target_update = 1
    batch_size = 32
    memory_size =  100000

    hyperparam_dict['gamma'] = gamma 
    hyperparam_dict['max_steps'] = max_steps 
    hyperparam_dict['max_episodes'] = max_episodes
    hyperparam_dict['epsilon'] = epsilon
    hyperparam_dict['epsilon_ending'] =eps_end 
    hyperparam_dict['epsilon_decay']= eps_dec 
    hyperparam_dict['target_update']=target_update
    hyperparam_dict['batch_size']=batch_size 
    hyperparam_dict['memory_size'] =memory_size
    
agent_all = a.Agent(hyperparam_dict)

if algorithm == 'REINFORCE':
    ep_rewards, running_rewards, mean_rew, std_rew = ra.reinforce(max_steps, 
                                                                  max_episodes,
                                                                  env, 
                                                                  agent_all.agent, 
                                                                  render=False,
                                                                  mode='train', 
                                                                  chooser=None,
                                                                  log_interval=interval, 
                                                                  tb=tboard,
                                                                  whole_loop = True,
                                                                  agent_whole = agent_all)
    
    # Store the results
    #results_neural_network.append((ep_rewards, 'Episode Rewards', 0, 'dash'))
    #results_neural_network.append((running_rewards, 'Running Rewards', 0, 'regular'))
    results_neural_network.append((running_rewards, 'Running Reward', 4, 'regular', ep_rewards))
    results_neural_network_meanstd.append((mean_rew, 'Std/Mean Reward Evaluation', 2, 'regular',std_rew ))
    if baseline:
        results_neural_network.append((agent_all.agent.baseline, 'Baseline', 1, 'regular'))
        results_neural_network_meanst.append((agent_all.agent.baseline, 'Baseline', 1, 'regular'))
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
    
    
if algorithm == 'DQN':
    ep_rewards, running_rewards, eps_history = qlt.q_learning(env, 
                                                              agent_all.agent, 
                                                              log_interval = 100,
                                                              whole_loop = True,
                                                              agent_whole = agent_all)
    
    print('Total steps: ' + str(agent_all.agent.steps_done))
    # Calculate the total time of execution
    print('The total time to compile '+str(max_episodes)+' episodes for a maximum of '+ str(max_steps) + ' steps is: ' + str((time.time() - start_time)/60))
    
    # Plot
    lists = []
    #lists.append((ep_rewards, 'Episode Rewards', 0, 'dash'))
    lists.append((running_rewards, 'Running Rewards', 4, 'regular', ep_rewards))
    lists.append((eps_history, 'Epsilon History', 1, 'regular'))
    #lists.append((avg_score, 'Average Rewards', 0))
    plot.multi_plot(lists, hyperparam_dict, 1, 'DQN')
    
    # Saving the hyperparameters dictionary into a file
    hyperparam_dict['total_time']= (time.time() - start_time)/60
    util.save_dict_txt(hyperparam_dict)


