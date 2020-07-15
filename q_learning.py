import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from itertools import count
import neural_network_policy as N
import replay as R
import random
import time

class DQN_Agent(object):
    def __init__(self, h):
        # Setting the Hyperparameters
        self.h = h
        self.n_actions = h['actions']
        self.n_states = h['states']
        self.GAMMA = h['gamma']
        self.lr = h['learning_rate']
        
        # Peculiarities of DQN
        self.EPS_START = h['epsilon']
        self.EPS_END = h['epsilon_ending']
        self.EPS_DECAY = h['epsilon_decay']
        self.TARGET_UPDATE = h['target_update']
        self.mem_size = h['memory_size']
        self.BATCH_SIZE = h['batch_size']
        if ('evaluation_interval' in self.h.keys()):
            self.EVALUATE_MODE = h['evaluation_interval']
        self.action_space = [i for i in range(self.n_actions)]
        
        # Instantiate the Networks
        self.policy_net = N.NeuralNetworkPolicy(self.n_actions, self.n_states)
        self.target_net = N.NeuralNetworkPolicy(self.n_actions, self.n_states)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Instantiate the optimizer, the memory object, and the loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()
        self.memory = R.ReplayMemory(self.mem_size, self.n_states, self.n_actions)
        self.steps_done = 0
        self.eps_threshold = 0
        # Storing list for the epsilon over the time
        self.eps_history = []    
        
    def eps_decay_linear_eq(self):
        # The equation is thought on a given number of iterations. 
        # It doesnt work good with done mode breaking of the steps loop
        # The linear decay idea is taken from Mnih 2013, Playing Atari with Deep Reinforcement Learning
        if (self.h['environment'] == 'CartPole-v0'):
            x =1500
        else:
            x = (self.h['max_steps']*self.h['max_episodes'])*3/4
        if self.steps_done >= x:
            self.eps_threshold =  self.EPS_END
        else: 
            self.eps_threshold = (((self.steps_done - 0)*(self.EPS_END - self.EPS_START))/ (x - 0))+self.EPS_START
        return self.eps_threshold
    
    def eps_decay_expo_eq(self):
        # The exponential decay idea comes from the pytorch official tutorial:
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # This equation approach comes from:
        # https://github.com/philtabor/Youtube-CodeRepository/blob/master/ReinforcementLearning/DeepQLearning
        #self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.eps_threshold = self.eps_threshold*self.EPS_DEC if self.eps_threshold > self.EPS_MIN else self.EPS_MIN
        return self.eps_threshold

    def choose_eval_action(self, observation, chooser):
        if chooser == 0:
            action = np.random.choice(self.action_space)
        if chooser == 1:
            actions = self.policy_net.forward(observation)
            action = torch.argmax(actions).item()
        return torch.tensor([action])

    def choose_action(self, observation):
        rand = np.random.random()
        actions = self.policy_net.forward(observation)
        # The exponential equation approach doesnt work good
        #self.eps_threshold = self.eps_decay_expo_eq()
        self.eps_threshold = self.eps_decay_linear_eq()
        self.steps_done += 1
        self.eps_history.append(self.eps_threshold)
        # Not using the no_grad() method, so the policy_net.forward is not included into the gradient computation
        if rand > self.eps_threshold:
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return torch.tensor([action])
        
    def select_action(self, state):
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1       
        self.eps_history.append(self.eps_threshold)
        if sample > self.eps_threshold:
            with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1,1)
        else:
            
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
        
    def optimize_model(self):
        # If the storing counter is higher than the batch_size, it passes the check
        if self.memory.position < self.BATCH_SIZE:
            return
        # Return an array 1xbatch_size of random exclusives indices of value between 0 and max_mem
        #batch = self.memory.sample(self.BATCH_SIZE)         
        # Return an array 1xbatch_size of random indices of value between 0 and max_mem: there are repeated values
        batch = self.memory.sample_redundance(self.BATCH_SIZE)
        # Loading of batch-th element of state, action and so on
        # state_batch[64 x 10]
        state_batch = self.memory.state_memory[batch]
        # action_batch[64 x 27]
        # action_batch example: [[1 0 0 0 .. 0] [0 0 0 1 ... 0] ...]
        action_batch = self.memory.action_memory[batch]
        # action_values: [0 1 .. 26]
        action_values = np.array(self.action_space, dtype = np.int32)
        # action_indices [64 ,] : [ 26 2 3 0 1 24 ...]
        action_indices = np.dot(action_batch, action_values)
        # reward_batch [1 x 64]
        reward_batch = self.memory.reward_memory[batch]
        # new_state_batch[64 x 10]
        new_state_batch = self.memory.new_state_memory[batch]
        # Array with zeros and ones depending on the done status
        terminal_batch = self.memory.terminal_memory[batch]
        # Changing shape and type to match in the next steps           
        state_batch = torch.from_numpy(state_batch).float().unsqueeze(1)
        new_state_batch = torch.from_numpy(new_state_batch).float().unsqueeze(1)
        action_batch = torch.from_numpy(action_indices).int().unsqueeze(1)
        reward_batch = torch.Tensor(reward_batch)
        terminal_batch = torch.Tensor(terminal_batch).float()
        # The DQN Notebook on pytorch works with batch elements like column tensor 
        # Using torch.cat() to avoid nested tensors
        state_batch_ = torch.cat([c for c in state_batch])
        new_state_batch_ = torch.cat([s for s in new_state_batch])
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch_)
        expected_state_action_values = state_action_values.clone()
                
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values  = torch.max(self.target_net(new_state_batch_),dim =1)[0]

        # Compute the expected Q values
        batch_index = np.arange(self.BATCH_SIZE, dtype=np.int32)
        expected_state_action_values[batch_index, action_indices]= reward_batch + self.GAMMA*next_state_values*terminal_batch

        # Compute Huber loss
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # Compute Mean Squared Error loss
        loss = self.loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def print_values(self, i_episode, episodic_reward, episodic_rewards, running_reward, log_interval, time_count):
        if (i_episode % log_interval == 0) \
        and (i_episode > 0) \
        and (self.h['environment'] == 'PlanarArmTeacher2Learner3-v1' or\
             self.h['environment'] == 'PlanarArmTeacher2Learner3-v2'):
            time_count = (time.time()-time_count)/60
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tEpsilon: {:.2f}\tTime: {:.2f}'\
                  .format(i_episode, episodic_reward, running_reward,  self.eps_threshold, time_count))
            time_count = time.time()
            
        # Print every log_interval Episode: Last Reward: Average Reward: Epsilon: Time:
        if self.h['environment'] == 'CartPole-v0':
            if (i_episode % 10 == 0) and (i_episode > 0):
                time_count = (time.time()-time_count)/60
                avg_score = np.mean(episodic_rewards[max(0, i_episode-10):(i_episode+1)])
                print('episode: ', i_episode,'score: ', episodic_reward, ' average score %.3f' % avg_score,\
                      'epsilon %.3f' % self.eps_history[i_episode])
                time_count = time.time()
            else:
                print('episode: ', i_episode,'score: ', episodic_reward)
        return time_count
        

def q_learning(env, DQN_Agent, log_interval = 100, whole_loop = False, agent_whole = None):
    num_episodes = DQN_Agent.h['max_episodes']
    num_steps = DQN_Agent.h['max_steps']
    running_rewards = []
    episodic_rewards = []
    # Lists to store mean and standard deviation for evaluation plotting 
    mean_eval_rewards = []
    std_eval_rewards = []    
    running_reward = None
    time_count = time.time()
    eps_history_avg = []
    for i_episode in count(1):
    # Initialize the environment and state
        
        if whole_loop:
            state = agent_whole.vision_module(env, 'reset', None)
            state = agent_whole.join_observations(state)
        else:
            state = env.reset()
            
        episodic_reward = 0
        t = 0
        #done = False       

        for i in range(0, num_steps):
            # Select and perform an action
            action = DQN_Agent.choose_action(torch.Tensor([state]))
            #action = DQN_Agent.select_action(torch.Tensor([state]))
            
            if whole_loop:
                state_, reward, done = agent_whole.vision_module(env, 'step', action)
                state_ = agent_whole.join_observations(state_)
            
            else:
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
                         
        # Evaluation Loop
        if ('evaluation_interval' in DQN_Agent.h.keys()):
            if (i_episode % DQN_Agent.EVALUATE_MODE == 0):
            #Recall the reinforce algorithm without exploration select-action algorithm
                _, _, average_reward_episode, std_reward_episode = eval_q_learning(env,
                                                                                   DQN_Agent,
                                                                                   chooser= 1,
                                                                                   whole_loop = whole_loop, 
                                                                                   agent_whole = agent_whole)  
                
                #Storing the data into lists for plotting
                mean_eval_rewards.append(average_reward_episode)
                std_eval_rewards.append(std_reward_episode)        
            
        # Stopping criteria
        if i_episode >= num_episodes:
            print('Max episodes exceeded, quitting.')
            break
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % DQN_Agent.TARGET_UPDATE == 0:
            DQN_Agent.target_net.load_state_dict(DQN_Agent.policy_net.state_dict())  
                
    # Save the data in the Q array
    DQN_Agent.policy_net.save(DQN_Agent.h['name'])
    
    return episodic_rewards, running_rewards, eps_history_avg, mean_eval_rewards, std_eval_rewards


def eval_q_learning(env, DQN_Agent, chooser=1, whole_loop = False, agent_whole = None):
    num_episodes_eval = DQN_Agent.h['max_episodes_eval']
    num_steps_eval = DQN_Agent.h['max_steps_eval']
    running_rewards = []
    episodic_rewards = []
    running_reward = None
    time_count = time.time()
    for i_episode in count(1):
    # Initialize the environment and state
        if whole_loop:
            state = agent_whole.vision_module(env, 'reset', None)
            state = agent_whole.join_observations(state)
        else:
            state = env.reset()
        
        episodic_reward = 0
        #done = False       

        for i in range(0, num_steps_eval):
            # Select and perform an action
            action = DQN_Agent.choose_eval_action(torch.Tensor([state]), chooser)
            if whole_loop:
                state_, reward, done = agent_whole.vision_module(env, 'step', action)
                state_ = agent_whole.join_observations(state_)
            
            else:
                state_, reward, done, _ = env.step(action.item())
            # Storing the reward
            reward = torch.tensor([reward])
            episodic_reward += reward.item()
            
            # Move to the next state
            state = state_

            if done:
                break
                                
        # Update the running reward: applying a exponential moving average 
        if running_reward is None:
            running_reward = episodic_reward
        else:
            running_reward = 0.05 * episodic_reward + ( 1- 0.05) * running_reward
            
        # Appending of the values to plot
        episodic_rewards.append(episodic_reward)
        running_rewards.append(running_reward)
                # Stopping criteria
    
        if i_episode >= num_episodes_eval:
            break                     
    run = np.array(running_rewards)
    ep = np.array(episodic_rewards)
    #print('ep'+ str(ep))

    #Compute the mean value of the obtained rewards	
    average_reward_running = run.mean()
    average_reward_episode = ep.mean()     
    #print('average '+ str(average_reward_episode))

    #Compute the standard deviation of the obtained rewards
    std_reward_running = run.std()
    std_reward_episode = ep.std()
    #print('std '+ str(std_reward_episode))
    
    return average_reward_running, std_reward_running, average_reward_episode, std_reward_episode

