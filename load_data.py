import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, x):
        self.x = torch.from_numpy(x).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Select sample
        x = self.x[index]

        # Load data and get label
        x = self.x[index]
        y = 0

        return x, y
    

def init_dataset_planar(batch_size, dataset_file_path ="learner_teacher_10k.npz", shuffle_bool=True, drop_last_bool=True):
    #Already existing dataset: see gen_data.py
    np_image_data = None
    img_size = None
    
    if os.path.isfile(dataset_file_path):
        # Load the images as a numpy array
        np_image_data = np.load(dataset_file_path)['array']
        print(f"Number of images: {np_image_data.shape[0]}, shape of an image: {np_image_data[0].shape}")
        
    torch_image_data = CustomDataset(np_image_data)
    #It could be an issue: train and test dataset are identical, only the positions change
    image_dataloader = DataLoader(torch_image_data, batch_size, shuffle=shuffle_bool, drop_last=drop_last_bool)
    if np_image_data[0][0].shape == np_image_data[0][1].shape:
        img_size=np_image_data[0][0].shape[0]
    else:
        print('Warning: The images are not squared')
    print(img_size)    
    return image_dataloader, img_size
    