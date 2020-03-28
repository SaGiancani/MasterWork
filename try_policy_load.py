import gym
import gym_control_model.planararm.register_env
import neural_network_policy as NeuralNetworkPolicy
import reinforce_algorithm as ra
import plot

import time

import numpy as np 
import torch

# Define a seed and other hyper parameters for reproducability
my_seed = 42
max_steps = 20
max_episodes = 50

# And set this seed in gym and torch
torch.manual_seed(my_seed)
np.random.seed(my_seed)
env = gym.make("PlanarArmTeacher2Learner3-v1")
env.seed(my_seed)
filename = 'Wed Mar 25 15_20_17 2020'
# Load the trained policy network
policy = NeuralNetworkPolicy.NeuralNetworkPolicy.load(filename+'_model.pt')

# Creation of a dictionary with hyper_params    
hyperparam_dict = {'name': filename, 'gamma':0.9, 'learning_rate':1e-3, 'max_steps':max_steps, 'max_episodes':max_episodes, 'my_seed':my_seed, 'actions': 27, 'states': 10, }

# Creation of an Agent object
agent = ra.REINFORCE_Agent(hyperparam_dict, policy = policy)

#Define a time counter
start_time = time.time()

print('++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++')
print('MODEL BASED SELECTION RENDER')
print('++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++')

# Evaluate the trained policy for a 100 episodes
policy_eval_ep_rewards, policy_eval_running_rewards, actions_list = ra.reinforce(max_steps, max_episodes, env, agent, render=True, mode='eval', chooser=1, log_interval=100)
#policy_eval_ep_rewards, policy_eval_running_rewards, actions_list = ra.eval_trained_policy(hyperparam_dict['max_episodes'], hyperparam_dict['max_steps'], env, policy, 1, render=True)

#Print
print('++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++')
print('RANDOM SELECTION RENDER')
print('++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++')

# Calculate the time of evaluation rendering
print('The total time to render '+str(max_episodes)+' episodes for a maximum of '+ str(max_steps) + ' steps is: ' + str((time.time() - start_time)/60))

time.sleep(5)   

# Evaluate the random policy for a 100 episodes
random_eval_ep_rewards, random_eval_running_rewards, action_list  = ra.reinforce(max_steps, max_episodes, env, agent, render=False, mode='eval', chooser=0, log_interval=100)
#random_eval_ep_rewards, random_eval_running_rewards, action_list = ra.eval_trained_policy(hyperparam_dict['max_episodes'], hyperparam_dict['max_steps'], env, policy, 0, render=False)

# Calculate the total time of random rendering
print('The total time to render '+str(max_episodes)+' episodes for a maximum of '+ str(max_steps) + ' steps is: ' + str((time.time() - start_time)/60))

# Custom method for plotting
plot.plot_reward(random_eval_ep_rewards, policy_eval_ep_rewards, 'Random Episode Rewards', 'Evaluated Model Episode Rewards', hyperparam_dict, num=4)
print(str(actions_list))
