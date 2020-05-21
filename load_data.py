import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, x, offset, length):
        self.x = torch.from_numpy(x).float()
        self.offset = offset
        self.length = length
        assert len(self.x)>=offset+length, Exception("Parent Dataset not long enough")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Select sample
        x = self.x[index+self.offset]

        # Load data and get label
        x = self.x[index+self.offset]
        y = 0
        return x, y
    

def init_dataset_planar(batch_size, 
                        val_share=0.1, 
                        dataset_file_path ="learner_teacher_10k.npz", 
                        shuffle_bool=True, drop_last_bool=True):

    #Already existing dataset: see gen_data.py
    np_image_data = None
    img_size = None
    
    if os.path.isfile(dataset_file_path):
        # Load the images as a numpy array
        np_image_data = np.load(dataset_file_path)['array']
        print(f"Number of images: {np_image_data.shape[0]}, shape of an image: {np_image_data[0].shape}")
    
    # Translation of the dataset type: numpy to torch
    val_offset = int(len(np_image_data)*(1-val_share))
    train_ds = CustomDataset(np_image_data, 0, val_offset)
    val_ds = CustomDataset(np_image_data, val_offset, len(np_image_data)-val_offset)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=shuffle_bool, drop_last=drop_last_bool)
    val_loader = DataLoader(val_ds, batch_size, shuffle=shuffle_bool, drop_last=drop_last_bool)

    if np_image_data[0][0].shape == np_image_data[0][1].shape:
        img_size=np_image_data[0][0].shape[0]
    else:
        print('Warning: The images are not squared')
        
    print(img_size)  
    print(len(train_loader))
    print(len(val_loader))
    
    return train_loader, val_loader, img_size
