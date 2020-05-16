import numpy as np

import agent_cae as ag_cae
import utilities as u

import torch
import torch.optim as optim
import plot
import random

import time

# Start a time counter
start_time = time.time()

# Print the current time
print(time.asctime(time.localtime(time.time())))

# Hyperparameters
FILENAME = str(time.asctime(time.localtime(time.time())))
RANDOM_SEED = 100
BATCH_SIZE = 128
DROP_LAST = True
NUM_WORKERS = 1
PIN_MEMORY = True
NUM_EPOCHS = 10
MAX_STEPS = 10
LR_PURE = 1e-5
PATH = "learner_teacher_10k.npz"
TEST_BATCH_NUM = 5
INIT = 'standard'
#INIT = 'xavier_normal'
#INIT = 'xavier_uniform'

# Initialize the seed
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check if cuda is available: else use cpu
cuda_available, device = u.check_cuda()

LOADER_KWARGS = {'num_workers': NUM_WORKERS,
                 'pin_memory': PIN_MEMORY} if cuda_available else {}

# Initialize the hyperparameters dictionary
hyperparam_dict = {'name': FILENAME, 
                   'learning_rate':LR_PURE, 
                   'steps_for_each_image':MAX_STEPS, 
                   'max_episodes':NUM_EPOCHS, 
                   'batch_size':BATCH_SIZE, 
                   'my_seed':RANDOM_SEED, 
                   'path_dataset':PATH,
                   'device': device,
                   'test_batch_num': TEST_BATCH_NUM,
                   'initialization': INIT
                   }

# Initialize the agent
agent = ag_cae.CAE_Agent(hyperparam_dict)
agent.train()

# Displaying the images over the epochs
plot.plot_imgs([(np.array(agent.img_array), None)], 
               len(agent.img_array), 
               'Output During Training: Pure conv with '+INIT,
               agent.img_size, FILENAME, 0)

# Batch of images for evaluation (from the train dataset)
imgs = agent.build_batch()

# Evaluate the obtained model
out, inp = agent.evaluation('model', imgs)
out = (out, 'Pure Conv')
inp = (inp, 'Original')

# Evaluate the code:
out_enc, _ = agent.evaluation('encoder', imgs)
outcome, _ = agent.evaluation('decoder', out_enc)
outcome = (outcome, 'Reconstruction')

# Plot the images obtained from the pre-trained model: evaluation and reconstruction
lis= []
lis.append(inp)
lis.append(out)
lis.append(outcome)
plot.plot_imgs(lis, 5, 'Evaluation Pure Conv Model', agent.img_size, FILENAME, 1)

# Plot of the loss across the epochs 
lists = []
lists.append((agent.epoch_losses, 'MSE Loss', 0, 'regular'))
plot.multi_plot(lists, hyperparam_dict, 1, "Only Convolutional Layers  lr: "+ str(LR_PURE), x='Epochs', y='Loss')

# Saving the hyperparameters dictionary into a file
hyperparam_dict['total_time']= (time.time() - start_time)/60
#u.save_dict_txt(hyperparam_dict)
print(hyperparam_dict)