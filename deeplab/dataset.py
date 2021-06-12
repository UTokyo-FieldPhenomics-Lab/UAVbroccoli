import os

import torch
import torch.nn as nn
from torchvision import  transforms

import numpy as np
from PIL import Image
import imageio

class Broccoli(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, size=1500, transforms=transforms.ToTensor()):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.data_type = data_type
        self.img_list = os.listdir(f'{img_dir}/{data_type}')
        self.mask_list = os.listdir(f'{mask_dir}/{data_type}')
        self.transforms = transforms
        
    def readImage(self, img_id):
        
        img = Image.open(img_id)
        transform = transforms.Resize((self.size, self.size))
        return transform(img)
    
    def __len__(self) -> int:
        return len(self.img_list)
    
    def __getitem__(self, index):
        
        img_id = self.img_list[index]
        
        img = self.readImage(f'{self.img_dir}/{self.data_type}/{img_id}')
        # print(f'{self.img_dir}/{self.data_type}/{img_id}')
        mask = self.readImage(f'{self.mask_dir}/{self.data_type}/{img_id}').convert("L")
        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)
        sample = {"image": img, "mask": mask}
                 
        return sample 
        
        
class Prediction(torch.utils.data.Dataset):
    def __init__(self, img_dir, size=256, transforms=transforms.ToTensor()):
        super().__init__()
        self.img_dir = img_dir
        self.size = size
        self.img_list = os.listdir(f'{img_dir}')
        self.transforms = transforms
        
    def readImage(self, img_id):
        # img = Image.open(img_id)
        img = imageio.imread(img_id)
        if img_id.endswith('tiff'):
            img = img[:, :, :3]
        img = Image.fromarray(img)
        transform = transforms.Resize((self.size, self.size))
        return transform(img)
    
    def __len__(self) -> int:
        
        return len(self.img_list)
    
    def __getitem__(self, index):
        
        img_id = self.img_list[index]
        
        img = self.readImage(f'{self.img_dir}/{img_id}')

        if self.transforms:
            img = self.transforms(img)
        if img_id.endswith('tiff'):
            img_id.replace('tiff', 'jpg')
        return img, img_id