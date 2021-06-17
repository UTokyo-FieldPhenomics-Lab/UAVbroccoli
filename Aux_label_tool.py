from genericpath import exists
import os
import random

import json
import numpy as np
from numpy.core.defchararray import asarray
from numpy.lib.polynomial import poly
import pandas as pd
from PIL import Image
import imageio
from skimage.transform import resize

import matplotlib.pyplot as plt
from torch._C import dtype
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

from deeplab.model import createDeepLabv3
from deeplab.dataset import Broccoli
from deeplab import args as arg
from utils.mask2polygon import mask2polygon
from utils.labelme2coco import labelme2json

ROOT = 'I:/Shared drives/broccoliProject/'
ROOT_ON_RAW_PATH = 'I:/Shared drives/broccoliProject/11_labelme_json/root_on_raw.json/'
JSON_PATH = 'I:/Shared drives/broccoliProject/11_labelme_json/json'
MASK_PATH = './deeplab/test/masks/'
os.makedirs(MASK_PATH, exist_ok=True)

device = 'cuda' if torch.cuda.is_available==True else 'cpu'

model = createDeepLabv3(pretrained=False)
model.to(device)

model_weight = torch.load('./deeplab/checkpoints/best_model.tar')
model_weight = model_weight["model_state_dict"]

model.load_state_dict(model_weight, strict=False)

def predict_batch(model, batch):
    """[summary]

    Args:
        model ([model]): [deeplab model]
        img ([list]): [batch of images]

    Returns:
        [type]: [description]
    """    
    with torch.no_grad():
        
        batch.to(device)
        # print(img)
        pred = model(batch)

        masks = pred['out'].permute(0, 2, 3, 1).detach().cpu().numpy()
        # images = batch.permute(0, 2, 3, 1).detach().cpu().numpy()
        # print(masks[0])
        # masks[masks >= 0.5] = 10
        # masks[masks < 0.5] = 255
        
        masks = masks.argmax(3).reshape(-1, 128, 128, 1)
        
        return np.array(masks, dtype=np.uint8)


def one_image(imageName, map_label, base=60):
    transform1 = transforms.Compose([
        transforms.Resize((arg.im_size, arg.im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # transform2 = 
    
    with open(map_label, 'r', encoding='utf-8') as f:
        # lines = f.readlines()
        label_data = json.load(f)
        
    imagePath = label_data[imageName]['imagePath']
    imagePath = ROOT + imagePath[40:]
    
    # print(imagePath)
    img = imageio.imread(imagePath)
    h, w, _ = img.shape
    one_mask = np.zeros((h, w, 1), dtype=np.uint8)
    points = label_data[imageName]['points']
    
    # convert to numpy array
    points = np.array(points, dtype=np.int32)
    # calcualte x0, y0, x1, y1
    y0 = points[:, 1] - base
    x0 = points[:, 0] - base
    y1 = points[:, 1] + base
    x1 = points[:, 0] + base
    # avoid minus number
    y0[y0 < 0] = 0
    x0[x0 < 0] = 0
    y1[y1 > h] = h
    x1[x1 > w] = w
    
    bboxs = np.array(list(zip(y0, x0, y1, x1)))
    # get sub images
    batch = []
    for y_0, x_0, y_1, x_1 in bboxs:
        sub_img = img[y_0:y_1, x_0:x_1, :3]
        sub_img = transform1(Image.fromarray(sub_img))
        sub_img = np.array(sub_img)
        batch.append(sub_img)
    batch = np.array(batch)
    batch = torch.from_numpy(batch)
    # print(batch.shape)
    # batch = torch.tensor(batch).permute((0, 3, 1, 2))
    
    # print(batch)
    masks = predict_batch(model, batch)
    # masks = np.array(masks, dtype=np.uint8)
    #paste back
    for box, mask in zip(bboxs, masks):
        # print(np.unique(mask))
        y0, x0, y1, x1 = box
        # print(mask, mask.shape) 
        mask = transforms.Resize((y1-y0, x1-x0))(torch.tensor(mask).permute((2,0,1)))*255
        mask[mask > 50] = 255
        mask[mask <= 50] = 0
        

        mask = np.asarray(mask).transpose((1,2,0))
        
        # print(mask.shape)
        one_mask[y0:y1, x0:x1, :] = mask
    imageio.imsave('./test.png', one_mask)
    return mask2polygon(one_mask[:, :, 0])
    
    # imageio.imwrite(os.path.join(MASK_PATH, name), mixed)
    
def Aux_label(json_path):
    """[summary]

    Args:
        dir_path ([string]): [path to target image folder]

    Returns:
        [list]: [list of cropped images with shape (n, c, w, h)]
    """    
    with open(json_path, 'r', encoding='utf-8') as f:
        labelme_json = json.load(f)
    
    img_path = labelme_json['imagePath']
    # print(img_path)
    img_name = img_path.split('\\')[-1]
    project_name = img_path.split('\\')[-2]
    map_label = f'{ROOT_ON_RAW_PATH}{project_name}.json'
    # print(map_label)
    print(f"start processing on: {json_path}")
    polygons = one_image(img_name, map_label=map_label)
    print("Prediction finished")
    
    for polygon in polygons:  
        coords = {
            "label": "broccoli",
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_json['shapes'].append(coords)
    
    new_name = json_path.replace('blank', 'aux')
    print(f"saving result to {new_name}")
    with open(json_path, 'w+') as f:
        f.write(json.dumps(labelme_json,indent=1))
    os.rename(json_path, new_name)
    print("result saved")

def Random_select_aux(number):
    json_list = [f'{JSON_PATH}/{entry.name}' for entry in os.scandir(JSON_PATH) if ((entry.name.startswith('blank')) & (entry.name.endswith('.json')))]
    # print(json_list)
    selected = random.sample(json_list, number)
    print(f'{number} files selected:')
    print(selected)
    for item in selected:
        Aux_label(item)
    
if __name__ == "__main__":
    # pass
    Random_select_aux(1)