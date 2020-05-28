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
NUM_EPOCHS = 500
IMAGES_DURING_TRAIN = 10
# Learning rate and parameters for lr_Scheduler
LR = 1e-4
GAMMA = 0.995
#MODE_DEC = 'EXP'
MODE_DEC = 'CYCLIC'
LR_SCHEDULER = True
#LR_SCHEDULER = False
#GAMMA = 0.99
if MODE_DEC == 'EXP' and LR_SCHEDULER:
    BASE_LR = None
    MAX_LR = None
    STEP_UP_TRIANGLE = None
    STEP_DOWN_TRIANGLE = None 
    MODE_SCHEDULER = None
    
if MODE_DEC == 'CYCLIC' and LR_SCHEDULER:
    BASE_LR = 5e-5
    MAX_LR = 5e-4
    STEP_UP_TRIANGLE = NUM_EPOCHS/10
    STEP_DOWN_TRIANGLE = None 
    MODE_SCHEDULER = 'triangular'
    #MODE_SCHEDULER = 'exp_range'

TEST_BATCH_NUM = 5
#Kind of initialization
#INIT = 'standard'
INIT = 'xavier_normal'
#INIT = 'xavier_uniform'

# Architecture mode
#MODE = ('pure', None)
# Path for the dataset
#PATH = "learner_teacher_10k.npz"

# Architecture mode 128x128
MODE = ('hybrid', 128)
# Path for the dataset
#PATH = "planar_arm_gray_128_20k.npz"
PATH = "learner_teacher_10k.npz"

# Architecture mode 64x64
#MODE = ('hybrid', 64)
# Path for the dataset
#PATH = "planar_arm_bin_64_100k-Copy1.npz"
#PATH = "planar_arm_gray_64_100k-Copy1.npz"

# Architecture mode 32x32
#MODE = ('hybrid', 32)
# Path for the dataset
#PATH = "planar_arm_bin_32_100k-Copy1.npz"
#PATH = "planar_arm_gray_32_100k-Copy1.npz"

# Percentage of the whole dataset dedicated to the Valutation
PERC_FOR_EVAL= 0.2
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
                   'lr_scheduler_mode': LR_SCHEDULER,
                   'cyclic_scheduler_mode': MODE_SCHEDULER,
                   'step_up_triangle': STEP_UP_TRIANGLE,
                   'step_down_triangle': STEP_DOWN_TRIANGLE
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