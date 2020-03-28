import numpy as np
import random


class ReplayMemory(object):
# The idea of the ReplayMemory object is taken from      https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    def __init__(self, capacity, inputs, outputs):
        self.n_input = inputs
        self.n_outputs = outputs
        self.capacity = capacity
        self.position = 0
        
        # I substitute the memory list with one np matrix for each gym element
        # I can delete the if check from the push method
        self.state_memory = np.zeros((self.capacity, self.n_input))
        self.new_state_memory = np.zeros((self.capacity, self.n_input))
        self.action_memory = np.zeros((self.capacity, self.n_outputs), dtype = np.uint8)
        self.reward_memory = np.zeros(self.capacity)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, terminal):     
        self.state_memory[self.position] = state
        actions = np.zeros(self.n_outputs)
        actions[action] = 1.0
        self.action_memory[self.position] = actions
        self.reward_memory[self.position] = reward
        # It's the done-status storing array
        self.terminal_memory[self.position] = 1-terminal
        self.new_state_memory[self.position] = state_
        self.position = (self.position + 1) % self.capacity
        #print('position index: ' + str(self.position))

    def sample(self, batch_size):
        # Return an array 1xbatch_size of random indices of value between 0 and max_mem without repeated values
        # max_mem max value of the possible index chosen: 
        # 1) max_mem = to the storing counter if it's less than the total size of the memory (1kk default)
        # 2) max_mem = to the total size of the memory (capacity)
        max_mem = self.position if self.position < self.capacity else self.capacity 
        return random.sample([i for i in range(max_mem)], batch_size)
    
    def sample_redundance(self, batch_size):
        # Return an array 1xbatch_size of random indices of value between 0 and max_mem without repeated values
        # max_mem max value of the possible index chosen: 
        # 1) max_mem = to the storing counter if it's less than the total size of the memory (1kk default)
        # 2) max_mem = to the total size of the memory (capacity)
        max_mem = self.position if self.position < self.capacity else self.capacity
        return np.random.choice(max_mem, batch_size)

