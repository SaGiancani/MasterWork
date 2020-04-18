import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import numpy as np
from itertools import count
import neural_network_policy as N
import baseline as b
import random
import time

class REINFORCE_Agent(object):
    def __init__(self, h, policy=None):
        # Setting the Hyperparameters
        self.h = h
        self.n_actions = self.h['actions']
        self.n_states = self.h['states']  
        self.GAMMA = self.h['gamma']        
        self.lr = self.h['learning_rate']
        print(self.lr)
        # Check on mode
        if 'evaluation_interval' in h.keys():
            self.EVALUATE_MODE = self.h['evaluation_interval']
        # Instantiate the Network
        if policy == None:
            self.policy = N.NeuralNetworkPolicy(self.n_actions, self.n_states)
            self.policy.train()
        else:
            self.policy = policy
            
        # Instantiate a baseline object. It's inspired to the replay system of the DQN approach    
        if ('baseline' in self.h.keys()) and (self.h['baseline']):
            self.value_function = N.NeuralNetworkPolicy(1, self.n_states)
            # It could work with lr/10: 1e-4 in both the optimizer or just on the value_function
            self.optimizer = optim.Adam(self.policy.parameters(), lr = self.lr)
            self.optimizer_value = optim.Adam(self.value_function.parameters(), lr = self.lr) 
            self.baseline = []
            self.running_base = None
            self.states = []
        
        else:
            # Instantiate the optimizer
            self.optimizer = optim.Adam(self.policy.parameters(), lr = self.lr)
        # Some utility counter
        self.counter=0

    def select_action(self, state):
        # Convert the state from a numpy array to a torch tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        # Get the predicted probabilities from the policy network
        action_scores = self.policy.forward(state)
        probs = F.softmax(action_scores, dim=1)
        # Sample the actions according to their respective probabilities
        m = Categorical(probs)
        action = m.sample()
        # Also calculate the log of the probability for the selected action
        self.policy.saved_log_probs.append(m.log_prob(action))
        # Return the chosen action
        return action.item(), probs
    
    def select_action_noexplo(self, state):
        # Convert the state from a numpy array to a torch tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        np.set_printoptions(precision=4, suppress=True)       
        # Get the predicted probabilities from the policy network
        action_scores = self.policy.forward(state)
        probs = F.softmax(action_scores, dim=1)
        # Sample the actions NOT according to their respective probabilities
        # I'm deleting the exploration component due to the probabilistic sampling
        #action = [int(m.argmax())]
        action = torch.argmax(action_scores).item()
        # Return the chosen action
        return action, probs 
    
    def select_random_action_eval(self, env):
        a = env.action_space.sample()
        return a

    def select_random_action_learn(self, env):
        np.set_printoptions(precision=4, suppress=True)
        # Impose the equal probability of all the actions
        probs = np.zeros((1, self.n_actions))
        probs[:] = 1/self.n_actions
        # Sample the actions according to their respective probabilities
        m = Categorical(torch.from_numpy(probs))
        action = env.action_space.sample()
        # Also calculate the log of the probability for the selected action
        self.policy.saved_log_probs.append(m.log_prob(torch.Tensor([action])).float())     
        return action, probs
    
    def select_action_greed(self, env, state):   
        # Choosing a randnum between 0 and 1
        randnum = np.random.random()
        if randnum < self.h['epsilon']:
            #Random action
            action, _ = self.select_random_action_learn(env)
            #print(action)
            return action       
        else:
            action, _ = self.select_action(state)
            return action

    
    def finish_episode(self, baseline_switch = False):
        # Variable for the current return
        policy_loss = []
        returns = []
        returns_base = []
        G = 0
        G_base = 0
                
        # If the baseline mode is actived this algorithm is used
        if baseline_switch and (self.states != None):
            # Go through the list of observed rewards and calculate the returns
            # gamma discount factor
            for r in self.policy.rewards[::-1]:
                # Undiscounted return function
                G_base = r + G_base
                returns_base.insert(0, G_base)
        
            # Convert the list of returns into a torch tensor
            returns_base = torch.tensor(returns_base)
            
            deltas = []
            base = []

            for s, G_base, log_prob  in zip(self.states, 
                                       returns_base, 
                                       self.policy.saved_log_probs):
                v = self.value_function.forward(s)
                base.append(v)
                delta = G_base - v
                deltas.append(-self.GAMMA*delta*v)
                policy_loss.append(-self.GAMMA*log_prob*delta)
    
            # To keep track of the baseline over the episode
            base_mean = torch.cat(base).mean()
            if self.running_base == None:
                self.running_base = base_mean
            else:
                self.running_base = 0.05 * base_mean + ( 1- 0.05) * self.running_base
            self.baseline.append(self.running_base)
            
            # w gradient (for the value_functin)
            self.optimizer_value.zero_grad()
            value_loss_w = torch.cat(deltas).mean()
            value_loss_w.backward(retain_graph=True)
            self.optimizer_value.step()
            
            # Theta gradient (for the policy)
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).mean()
            policy_loss.backward(retain_graph=True)
            self.optimizer.step()
            
            # Reset the saved rewards and log probabilities
            del self.states[:]
            del self.policy.rewards[:]
            del self.policy.saved_log_probs[:]
                    
        # Else the standard REINFORCE is used
        else:
            # Define a small float which is used to avoid divison by zero
            eps = np.finfo(np.float32).eps.item()
        
            # Go through the list of observed rewards and calculate the returns
            # gamma discount factor
            for r in self.policy.rewards[::-1]:
                G = r + self.GAMMA * G
                returns.insert(0, G)
    
            # Convert the list of returns into a torch tensor
            returns = torch.tensor(returns)
            
            # Here we normalize the returns by subtracting the mean and dividing
            # by the standard deviation. Normalization is a standard technique in
            # deep learning and it improves performance, as discussed in
            # http://karpathy.github.io/2016/05/31/rl/
            returns = (returns - returns.mean()) / (returns.std() + eps)
        
            # Here, we deviate from the standard REINFORCE algorithm as discussed above
            for log_prob, G in zip(self.policy.saved_log_probs, returns):
                policy_loss.append(-log_prob * G)
        
            # Reset the gradients of the parameters
            self.optimizer.zero_grad()
    
            # Compute the cumulative loss
            policy_loss = torch.cat(policy_loss).mean()
    
            # Backpropagate the loss through the network
            policy_loss.backward()
    
            # Perform a parameter update step
            self.optimizer.step()
        
            # Reset the saved rewards and log probabilities
            del self.policy.rewards[:]
            del self.policy.saved_log_probs[:]


def reinforce(max_steps, max_episodes, env, Agent, render=False, mode='train', chooser=None, log_interval=100, tb=None):
    if mode == 'train':
        select_action = Agent.h['select_action_algorithm']
        max_steps_eval = Agent.h['max_steps_eval']
        max_episodes_eval = Agent.h['max_episodes_eval']
        baseline_switch = Agent.h['baseline']
        
    # Counter and time counter
    time_count=time.time()
    
    # To track the reward across consecutive episodes (smoothed)
    running_reward = None
    
    # Lists to store the episodic and running rewards for plotting
    ep_rewards = list()
    running_rewards = list()
    actions_array = list()
    
    # Lists to store mean and standard deviation for evaluation plotting 
    mean_eval_rewards = list()
    std_eval_rewards = list()
     
    # Start executing an episode (here the number of episodes is unlimited)
    for i_episode in count(1):
        
        # Make an empty list to store the actions for the TensorBoardX
        #actions_prob = torch.zeros(hyperparam_dict['max_steps'], 27)
        #actions_array = torch.zeros((hyperparam_dict['max_steps']), 1)

        # Reset the environment
        state, ep_reward = env.reset(), 0
        # For each step of the episode
        for t in range(0, max_steps):
            Agent.counter +=1          
            
            if mode == 'train':
                if select_action == 'greed':
                    # Select an action using the greedy algorithm
                    action = Agent.select_action_greed(env, state)
            
                if select_action == 'regular':
                    # Select an action using the policy network
                    action, _ = Agent.select_action(state)
   
                #actions_prob[t][:]= probs
                actions_array.append(action)
                
                # Perform the action and note the next state and reward
                state, reward, done, _ = env.step(action)

                # Store the current reward
                Agent.policy.rewards.append(reward)
                
                if baseline_switch:
                    # Convert the state from a numpy array to a torch tensor
                    state_v = torch.from_numpy(state).float().unsqueeze(0)
                    #value = Agent.value_function.forward(state_v)
                    #print('state: ' + str(state_v))
                    #Agent.value_function.states.append(state_v)
                    Agent.states.append(state_v)
                    #baseline.store(reward)
            
            if mode == 'eval':
                if chooser == 1:
                    # Select an action using the policy network
                    action, _ = Agent.select_action_noexplo(state)
                    actions_array.append(action)
                    
                if chooser == 0:
                    action = Agent.select_random_action_eval(env)
                    actions_array.append(action)
                  
                # Perform the action and note the next state and reward
                state, reward, done, _ = env.step(action)
            
            if render:
                env.render()
                time.sleep(1)
                print('observation: {}, reward: {}'.format(state, reward))


            # Track the total reward in this episode
            ep_reward += reward
                                 
            if done:
                break
               
        # Update the running reward: applying a exponential moving average 
        if running_reward is None:
            running_reward = ep_reward
        else:
            running_reward = 0.05 * ep_reward + ( 1- 0.05) * running_reward
        
        # Store the rewards for plotting
        ep_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        
        if (mode == 'eval') and (i_episode>=max_episodes):
            return ep_rewards, running_rewards, actions_array
            
        if mode == 'train':
            actions_array_ = torch.Tensor(actions_array[-max_steps:]).float().unsqueeze(1)
            # Plot on TensorBoardx the occurrance of an action over time steps
            #tb.add_histogram('Actions Occurrance', actions_array_, i_episode)	        

            # Plot on TensorBoardx the mean of the probabilities of an action over the time steps
            #for i, val in enumerate(torch.mean(actions_prob, axis=0)):
            #    tb.add_scalar("act/%d" % (i,), val, i_episode)
               
            # Plot on TensorBoardx the rewards
            #for i in range(len(running_rewards)):
            #    tb.add_scalar('Reward running', running_reward, i_episode)
            #    tb.add_scalar('Episode Reward', ep_reward, i_episode )
        
            #tb.close()

            # Perform the parameter update according to REINFORCE
            Agent.finish_episode(baseline_switch)
            
            # Print check: print Episode, Last and Average reward every log_interval times
            if (i_episode % log_interval == 0)  and (i_episode > 0):
                time_count = (time.time()-time_count)/60
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tTime: {:.2f}'.format(
                i_episode, ep_reward, running_reward, time_count))
                time_count = time.time()
                
            # Evaluation Loop
            if (i_episode % Agent.EVALUATE_MODE == 0):
                #Recall the reinforce algorithm without exploration select-action algorithm
                _, _, average_reward_episode, std_reward_episode = eval_into_reinforce(max_steps_eval, 
                                                                                       max_episodes_eval,
                                                                                       env, Agent)      
                #Storing the data into lists for plotting
                mean_eval_rewards.append(average_reward_episode)
                std_eval_rewards.append(std_reward_episode)
                
            # Stopping criteria
            if i_episode >= max_episodes:
                print('Max episodes exceeded, quitting.')
                # Save the trained policy network
                Agent.policy.save(Agent.h['name'])
                print('Counter: ' + str(Agent.counter))
                return ep_rewards, running_rewards, mean_eval_rewards, std_eval_rewards

def eval_into_reinforce(max_steps, max_episodes, env, Agent):
    #Recall the reinforce algorithm with without exploration select-action algorithm
    policy_eval_ep_rewards, policy_eval_running_rewards, _= reinforce(max_steps,
                                                                      max_episodes,
                                                                      env, 
                                                                      Agent, 
                                                                      render=False,
                                                                      mode='eval',
                                                                      chooser = 1)

    run = np.array(policy_eval_running_rewards)
    ep = np.array(policy_eval_ep_rewards)
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
