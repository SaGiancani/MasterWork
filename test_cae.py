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
NUM_EPOCHS = 1000
IMAGES_DURING_TRAIN = 10
# Learning rate and parameters for lr_Scheduler
LR = 1e-3
GAMMA = 0.995
BASE_LR = 5e-4
MAX_LR = 5e-3
#BASE_LR = None
#MAX_LR = None
LR_SCHEDULER = True
#LR_SCHEDULER = False
# Path for the dataset
PATH = "learner_teacher_10k.npz"
TEST_BATCH_NUM = 5
#Kind of initialization
#INIT = 'standard'
INIT = 'xavier_normal'
#INIT = 'xavier_uniform'
# Architecture mode
#MODE = 'pure'
MODE = 'hybrid'
# Percentage of the whole dataset dedicated to the Valutation
PERC_FOR_EVAL= 0.1
# How many steps between a print and another
INTERVAL = 10

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
                   'learning_rate':LR,
                   'steps_for_each_image':IMAGES_DURING_TRAIN, 
                   'max_episodes':NUM_EPOCHS, 
                   'batch_size':BATCH_SIZE, 
                   'my_seed':RANDOM_SEED, 
                   'path_dataset':PATH,
                   'device': device,
                   'test_batch_num': TEST_BATCH_NUM,
                   'initialization': INIT,
                   'mode': MODE,
                   'percentage_of_dataset_for_eval': PERC_FOR_EVAL,
                   'interval_for_print': INTERVAL,
                   'lr_scheduler_mode': LR_SCHEDULER
                   }

# Check for the use of lr_scheduler: if Base and Max lr are not instantiated, the program use the simple optimizer 
if LR_SCHEDULER:
    if (BASE_LR != None) and (MAX_LR != None):
        hyperparam_dict['max_learning_rate'] = MAX_LR
        hyperparam_dict['base_learning_rate'] = BASE_LR
    hyperparam_dict['gamma'] = GAMMA
    

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
out = (out, str(MODE)+'Conv')
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
plot.plot_imgs(lis, 5, 'Evaluation ' + str(MODE) + ' Conv Model', agent.img_size, FILENAME, 1)

# Plot of the loss across the epochs 
lists = []
lists.append((agent.epoch_losses, 'Train', 0, 'regular'))
lists.append((agent.evaluated_losses, 'Validation', 0, 'regular'))

# Check for the titles setting
if LR_SCHEDULER:
    if ('base_learning_rate' in hyperparam_dict.keys()) and ('max_learning_rate' in hyperparam_dict.keys()):
        tit = str(MODE) + " Convolutional Layers witch Cycling lr decay"
        tit_ = "Cyclic Learning Rate decay over the epochs  Gamma: " + str(GAMMA)+" Base lr: "+str(BASE_LR)+" Max lr: "+ str(MAX_LR)
        plot_type = 0
        # Plot lr over the epochs only if lr_scheduler mode is on
        l= []
        l.append((agent.lr_list, 'Learning Rates Decay', plot_type, 'regular'))
        plot.multi_plot(l, hyperparam_dict, 2,
                        tit_,
                        x='Epochs', y='lr value' )
    else:
        tit = str(MODE) + " Convolutional Layers with exponential lr decay   lr from: " +str(LR)+ "  and Gamma: " + str(GAMMA)
        # See the multiplot method
        plot_type = 1
        lists.append((agent.lr_list, 'Learning Rates Decay', plot_type, 'regular' ))

else:
    tit = str(MODE) + " Convolutional Layers lr: "+ str(LR)
        
plot.multi_plot(lists, hyperparam_dict, 1, tit, x='Epochs', y='Loss')


# Saving the hyperparameters dictionary into a file
hyperparam_dict['total_time']= (time.time() - start_time)/60
#u.save_dict_txt(hyperparam_dict)
print(hyperparam_dict)