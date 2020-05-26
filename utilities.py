import torch
import torch.nn as nn
import torch.nn.functional as F


def save_dict_csv(hyperparam_dict):
    w = csv.writer(open(hyperparam_dict['name']+"_hyperParam.csv", "w"))
    for key, val in hyperparam_dict.items():
        w.writerow([key, val])
        
def save_dict_txt(hyperparam_dict):
    f = open(hyperparam_dict['name']+'/'+hyperparam_dict['name']+"_hyperParam.txt","w")
    f.write(str(hyperparam_dict) )
    f.close()
    
def check_cuda():
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    print('cuda_available: {}, device: {}'.format(cuda_available, device))
    return cuda_available, device

#Understanding the difficulty of training deep feedforward neural networks - Glorot, Xavier & Bengio, Y. (2010)
def xavier_init_uniform(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

def xavier_init_normal_(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()

#Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)        
def kaiming_init_normal_(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        m.bias.data.zero_()

def kaiming_init_uniform(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        m.bias.data.zero_() 