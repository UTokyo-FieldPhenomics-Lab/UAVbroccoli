import os

import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from PIL import Image
import imageio
from sklearn import model_selection
from pycocotools.coco import COCO

from tqdm import tqdm

class Broccoli(torch.utils.data.Dataset):
    def __init__(self, coco_label_path, size=256, transforms=transforms.ToTensor()):
        super().__init__()
        self.coco = COCO(coco_label_path)
        imgIds = self.coco.getImgIds()
        annIds = self.coco.getAnnIds()
        self.images = self.coco.loadImgs(imgIds)
        self.anns = self.coco.loadAnns(annIds)
        
        self.size = size
        self.transforms = transforms
        

    def get_sub(self, img, ann):
        
        h, w, _ =img.shape 
        transform = transforms.Resize((self.size, self.size))
        
        xmin = ann["bbox"][0]
        ymin = ann["bbox"][1]
        xmax = ann["bbox"][2] + ann["bbox"][0]
        ymax = ann["bbox"][3] + ann["bbox"][1]

        base = 50
        xmin = xmin - base if (xmin - base) >= 0 else 0
        ymin = ymin - base if (ymin - base) >= 0 else 0
        xmax = xmax + base if (xmax + base) <= h else h
        ymax = ymax + base if (ymax + base) <= w else w
        
        sub_image = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
        mask = self.coco.annToMask(ann) * 255.
        mask = mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        sub_image = Image.fromarray(sub_image)
        mask = Image.fromarray(mask)
        
        return transform(sub_image), transform(mask)
        
    def __len__(self) -> int:
        return len(self.anns)
    
    def __getitem__(self, index):
        ann = self.anns[index]
        image = self.coco.loadImgs([ann['image_id']])[0]
        img = imageio.imread(image['imagePath'])
        sub, mask = self.get_sub(img, ann)

        if self.transforms:
            sub = self.transforms(sub)
            mask = self.transforms(mask)
        sample = {"image": sub, "mask": mask}
                 
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