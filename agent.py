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
        self.model_cae = hybrid_32_cae.load('32bitSetModel.pt')
        self.q_agent = qlt.DQN_Agent(hyperparam_dict)
        self.r_agent = ra.REINFORCE_Agent(hyperparam_dict)
        # Note for the new gym environment
        #env = gym.make("PlanarArmTeacher2Learner3-v2")
        #cv.imshow('teacher',observation[1])
        #cv.waitKey(10)
        #cv.imshow('learner',observation[0])

def learning(env, Agent, log_interval = 100):
    num_episodes = Agent.h['max_episodes']
    num_steps = Agent.h['max_steps']
    img_size = Agent.h['img_size']
    img_binary = Agent.h['img_binary']
    dtype_num = Agent.h['type']
    
    running_rewards = []
    episodic_rewards = []
    running_reward = None
    time_count = time.time()
    eps_history_avg = []
    
    for i_episode in count(1):
    # Initialize the environment and state
        state = env.reset()
        learner, teacher = gen.generate_images_agent(state, img_size, img_binary, dtype_num)
        learner_neck = Agent.model_cae(learner)
        
        episodic_reward = 0
        t = 0
        #done = False       

        for i in range(0, num_steps):
            # Select and perform an action
            action = Agent.choose_action(torch.Tensor([state]))
            #action = DQN_Agent.select_action(torch.Tensor([state]))
            state_, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])
            next_state = state_
            
            # Store the transition in memory
            DQN_Agent.memory.store_transition(state, action, reward, next_state, done)

            # Storing the reward
            DQN_Agent.policy_net.rewards.append(reward.item())  
            episodic_reward += reward.item()
            
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            DQN_Agent.optimize_model()              
            
            if done:
                break
                                
            # +1 to the counter
            t +=1
                                                    
        # Update the running reward: applying a exponential moving average 
        if running_reward is None:
            running_reward = episodic_reward
        else:
            running_reward = 0.05 * episodic_reward + ( 1- 0.05) * running_reward
            
        # Appending of the values to plot
        episodic_rewards.append(episodic_reward)
        running_rewards.append(running_reward)
        eps_history_avg.append(np.mean(DQN_Agent.eps_history[-t:]))
        
        # If PlanarArm print every log_interval Episode: Last Reward: Average Reward: Epsilon: Time:
        # If CartPole print ever 10 episodes Episode, Score, Average Score, Epsilon, and every episode Episode, Score
        time_count = DQN_Agent.print_values(i_episode, episodic_reward,
                                            episodic_rewards, running_reward, 
                                            log_interval, time_count)
                         
        # Stopping criteria
        if i_episode >= num_episodes:
            print('Max episodes exceeded, quitting.')
            break
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % DQN_Agent.TARGET_UPDATE == 0:
            DQN_Agent.target_net.load_state_dict(DQN_Agent.policy_net.state_dict())  
                
    # Save the data in the Q array
    DQN_Agent.policy_net.save(DQN_Agent.h['name'])
    
    return episodic_rewards, running_rewards, eps_history_avg

        