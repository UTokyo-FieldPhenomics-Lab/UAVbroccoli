import os

import numpy as np
import pandas as pd
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from model import createDeepLabv3
from dataset import Broccoli, Prediction

class Interface:
    def __init__(self, model_weight, img_dir, out_dir, device):
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
        self.model.classifier = DeepLabHead(2048, 1)

        self.model.to(device)
        model_weight = torch.load(model_weight)
        model_weight = model_weight["model_state_dict"]
        
        self.model.load_state_dict(model_weight, strict=False)
        
        self.img = Prediction(img_dir=img_dir, size=1500)
        self.dataloader = DataLoader(
            self.img,
            batch_size=16,
            num_workers=4,
            shuffle=False
        )
        
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.device = device
    
    def mask2polygon(self):
        pass
    def pred(self):
              
        self.model.eval()
        with torch.no_grad():
            for img, img_id in tqdm(self.dataloader):
                img = img.to(self.device)
                # print(img)
                pred = self.model(img)

                img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
                pred_masks = pred['out'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # print(pred_masks[0])
                pred_masks[pred_masks >= 0.5] = 200
                pred_masks[pred_masks < 0.5] = 255
                
                for im, im_mask, im_name in zip(img, pred_masks, img_id):
                    # im = imageio.imread(im_mask[:, :, 0])
                    result = np.concatenate((im*255, im_mask), axis=2)
                    # print(np.unique(im_mask))
                    imageio.imwrite(os.path.join(self.out_dir, f'{im_name[:-4]}.png'), result.astype(np.uint8))
                    # imageio.imwrite(os.path.join(self.out_dir, f'{im_name[:-4]}.png'), im_mask.astype(np.uint8))
                    

if __name__ == "__main__":
    device = "cuda"
    import args
    interface = Interface('./checkpoints/full/best_model.tar', r'I:\Shared drives\broccoliProject\10_anotation_use\geotiff\broccoli_tanashi_5_20200514_P4M_10m', out_dir='./seg_result/broccoli_tanashi_5_20200514_P4M_10m', device=device) 
    interface.pred()