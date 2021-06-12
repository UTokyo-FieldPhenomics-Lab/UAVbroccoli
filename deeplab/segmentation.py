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

PROJECT_PATH = r'I:\Shared drives\broccoliProject\11_instance_seg\detect+bg'
PROJETC_NAME = 'broccoli_tanashi_5_20200514_P4M_10m'

IMG_PATH = os.path.join(PROJECT_PATH, PROJETC_NAME)
LABEL_PATH = os.path.join(PROJECT_PATH, PROJETC_NAME, 'labels')
device = 'cuda' if torch.cuda.is_available==True else 'cpu'

model = createDeepLabv3(pretrained=False)
model.to(device)

model_weight = torch.load('./checkpoints/full/best_model.tar')
model_weight = model_weight["model_state_dict"]

model.load_state_dict(model_weight, strict=False)


def read_labels()
    label_list = os.listdir(LABEL_PATH)
    bbox_pd = pd.DataFrame(columns=['img_id', 'xc', 'yc', 'w', 'h'])

    
    for label_txt in label_list:
        print(f"reading {label_txt}", end="\r")
        with open(f"{LABEL_PATH}/{label_txt}") as f:
            for l in f.readlines():
                _, xc, yc, w, h, _ = l.split(' ')

                bbox_pd.loc[len(bbox_pd),:] = [label_txt.replace('txt', 'tif'), 
                                            float(xc), float(yc),
                                            float(w), float(h)]
    bbox = bbox_pd.astype(np.float32)
    bbox['xc'] = bbox['xc'] * (grid_len + buffer_len)
    bbox['yc'] = bbox['yc'] * (grid_len + buffer_len)
    bbox['w']  = bbox['w'] * (grid_len + buffer_len)
    bbox['h']  = bbox['h'] * (grid_len + buffer_len)

    bbox['x0'] = bbox['xc'] - bbox['w'] / 2
    bbox['x1'] = bbox['xc'] + bbox['w'] / 2
    bbox['y0'] = bbox['yc'] - bbox['h'] / 2
    bbox['y1'] = bbox['yc'] + bbox['h'] / 2
    
    bbox = bbox.astype(np.uint16)
    
    return bbox

def read_batch_images(img, bboxs):
    batch_imgs = []
    transform = transforms.Resize((256, 256))
    for index, row in bboxs.iterrows():
        x0 = row['x0']
        x1 = row['x1']
        y0 = row['y0']
        y1 = row['y1']
        
        base = 50
        x0 = x0 - base if (x0 - base) >= 0 else 0
        y0 = y0 - base if (y0 - base) >= 0 else 0
        x1 = x1 + base if (x1 + base) <= 1500 else 1500
        y1 = y1 + base if (y1 + base) <= 1500 else 1500
        
        sub_img = img[y0:y1, x0:x1, 3]
        sub_img = Image.fromarray(sub_img)
        sub_img = transform(sub_img)
        sub_img = numpy.asarray(sub_img)
        
        batch_imgs.append(sub_img)
    batch_imgs = np.array(batch_imgs).transpose((0, 3, 1, 2))
    return batch_imgs

def predict_batch(model, img):
    with torch.no_grad():
        
        img = img.to(device)
        # print(img)
        pred = model(img)

        img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_masks = pred['out'].permute(0, 2, 3, 1).detach().cpu().numpy()
        # print(pred_masks[0])
        pred_masks[pred_masks >= 0.5] = 200
        pred_masks[pred_masks < 0.5] = 255
        
        return pred_masks


if __name__ == "__main__":
    img_ids = [img_id for img_id in os.listdir(IMG_PATH) if img_id.endswith('tif') or img_id.endswith('jpg')] 
    bboxs = read_labels()
    for img_id in img_ids:
        img_dir = os.path.join(IMG_PATH, img_id)
        img = imageio.imread(img_dir)
        bbox = bboxs[bboxs.img_id == img_id]
        
        batch_images = read_batch_images(img, bbox)
        masks = predict_batch(model, batch_images)
        
        for mask in masks
        
            for im, im_mask, im_name in zip(img, pred_masks, img_id):
            # im = imageio.imread(im_mask[:, :, 0])
            result = np.concatenate((im*255, im_mask), axis=2)
            # print(np.unique(im_mask))
            imageio.imwrite(os.path.join(out_dir, f'{im_name[:-4]}.png'), result.astype(np.uint8))
            # imageio.imwrite(os.path.join(out_dir, f'{im_name[:-4]}.png'), im_mask.astype(np.uint8))
        




out_dir = out_dir
os.makedirs(out_dir, exist_ok=True)