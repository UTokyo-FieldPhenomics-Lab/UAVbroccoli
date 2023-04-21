import os

import torch
import torch.nn as nn
from torchvision import transforms
# from pathlib import Path
from skimage.transform import resize
import numpy as np
from PIL import Image
import imageio
from sklearn import model_selection
from tqdm import tqdm
# from tqdm import tqdm

class Broccoli(torch.utils.data.Dataset):
    def __init__(self, coco, size=256, transforms=None):
        super().__init__()
        self.coco = coco
        imgIds = coco.getImgIds()
        annIds = coco.getAnnIds()
        self.images = coco.loadImgs(imgIds)
        self.anns = coco.loadAnns(annIds)
        
        self.size = size
        self.transforms = transforms

        self.train_x, self.train_y = self.load_data()
        
    def get_sub(self, img, mask, ann):
        h, w, _ =img.shape
        # transform = transforms.Resize((self.size, self.size))
        
        x_c = ann["bbox"][2]//2 + ann["bbox"][0]
        y_c = ann["bbox"][3]//2 + ann["bbox"][1]

        base = 75
        xmin = x_c - base if (x_c - base) >= 0 else 0
        ymin = y_c - base if (y_c - base) >= 0 else 0
        xmax = x_c + base if (x_c + base) <= w else w
        ymax = y_c + base if (y_c + base) <= h else h

        sub_image = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
        mask = mask[int(ymin):int(ymax), int(xmin):int(xmax)]

        sub_image = resize(sub_image, (self.size, self.size))
        mask = resize(mask*255, (self.size, self.size))
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0

        return sub_image, mask
        
    def load_data(self):
        # print("saving images into .npy")
        Imgs = []
        Masks = []
        for image in tqdm(self.images):
            # print(image['imagePath'])
            # print(image['imagePath'].split('_')[3][:4])
            # p = Path(image['imagePath'])
            imagePath = image['imagePath']
            img = imageio.imread(imagePath)
            if img.shape[-1] == 4:
                img = img[..., :3]
            id = image['id']
            anns_ids = self.coco.getAnnIds(imgIds = [id])
            anns = self.coco.loadAnns(ids=anns_ids)
            masks = self.coco.annToMask(anns[0])
            for i in range(len(anns)):
                name = self.coco.loadCats(anns[i]['category_id'])[0]['name']
                if name != 'broccoli':
                    continue
                masks = masks | self.coco.annToMask(anns[i])

            for ann in anns:
                sub_image, mask = self.get_sub(img, masks, ann)
                Imgs.append(sub_image)
                Masks.append(mask)

        return Imgs, Masks
        
    def __len__(self) -> int:
        return len(self.anns)
    
    def __getitem__(self, index):
        sub = self.train_x[index] * 255
        # print(sub)
        mask = self.train_y[index] * 255
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
    
        sub = sub.astype(np.uint8)
        mask = mask.astype(np.uint8)
        # print(sub, mask)
        if self.transforms:
            transformed = self.transforms(image=sub, mask=mask)
            # mask = self.transforms(mask)
            sub = transformed['image']
            mask = transformed['mask']
        
        return sub, mask