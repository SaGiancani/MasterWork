import gym
import numpy as np

import gym_control_model.planararm.register_env
import q_learning as qlt
import utilities as util

import torch
import torch.optim as optim
import plot

import time

# Start a time counter
start_time = time.time()

# Print the current time
print(time.asctime(time.localtime(time.time())))

# Initialize the gym environment with the seed
my_seed = 3333
np.random.seed(my_seed)
torch.manual_seed(my_seed)
#environment = 'CartPole-v0'
environment = 'PlanarArmTeacher2Learner3-v1'
env = gym.make(environment)
env.seed(my_seed)

# Define name and hyperparameters of the model
filename = str(time.asctime(time.localtime(time.time())))
n_joints = (2,3) 
epsilon = 1
gamma = 0.99
lr = 1e-3
batch_size = 32
max_episodes, max_steps = 2500 , 200
memory_size =  100000
eps_end = 0.01
eps_dec = 0.996
target_update = 1
actions = env.action_space.n
if environment=='PlanarArmTeacher2Learner3-v1':
    states = 10
if environment=='CartPole-v0':
# The following equation doesnt work very well with CartPole
    states = 4
# Storing the hyparameters into a dictionary
hyperparam_dict = {'name': filename, 'n_joints':n_joints,'epsilon': epsilon,
                   'epsilon_ending':eps_end, 'epsilon_decay': eps_dec, 'gamma':gamma, 
                   'learning_rate':lr, 'max_steps':max_steps, 'max_episodes':max_episodes, 
                   'target_update':target_update,'batch_size':batch_size, 
                   'memory_size': memory_size, 'my_seed':my_seed, 
                   'actions':actions, 'states':states, 'environment': environment 
                  }

# Initialize the algorithm
agent = qlt.DQN_Agent(hyperparam_dict)
ep_rewards, running_rewards, eps_history = qlt.q_learning(env, agent, log_interval = 100)

print('Total steps: ' + str(agent.steps_done))
# Calculate the total time of execution
print('The total time to compile '+str(max_episodes)+' episodes for a maximum of '+ str(max_steps) + ' steps is: ' + str((time.time() - start_time)/60))

# Plot
lists = []
lists.append((ep_rewards, 'Episode Rewards', 0, 'dash'))
lists.append((running_rewards, 'Running Rewards', 0, 'regular'))
lists.append((eps_history, 'Epsilon History', 1, 'regular'))
#lists.append((avg_score, 'Average Rewards', 0))
plot.multi_plot(lists, hyperparam_dict)

# Saving the hyperparameters dictionary into a file
hyperparam_dict['total_time']= (time.time() - start_time)/60
util.save_dict_txt(hyperparam_dict)
