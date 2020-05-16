import torch

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