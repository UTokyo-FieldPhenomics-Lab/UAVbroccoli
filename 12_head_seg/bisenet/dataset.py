import os

import torch
import torch.nn as nn
from torchvision import transforms

from skimage.transform import resize
import numpy as np
from PIL import Image
import imageio
from sklearn import model_selection
from tqdm import tqdm
# from tqdm import tqdm

ROOT = 'G:/Shared drives/broccoliProject/13_roi_on_raw'

class Broccoli(torch.utils.data.Dataset):
    def __init__(self, coco, size=256, save_data=True, transforms=None):
        super().__init__()
        self.coco = coco
        imgIds = coco.getImgIds()
        annIds = coco.getAnnIds()
        self.images = coco.loadImgs(imgIds)
        self.anns = coco.loadAnns(annIds)
        
        self.size = size
        self.transforms = transforms
        if save_data:
            self.save2npy()
        self.train_x, self.train_y = self.load_npy()
        
    def get_sub(self, img, ann):
        
        h, w, _ =img.shape
        # transform = transforms.Resize((self.size, self.size))
        
        xmin = ann["bbox"][0]
        ymin = ann["bbox"][1]
        xmax = ann["bbox"][2] + ann["bbox"][0]
        ymax = ann["bbox"][3] + ann["bbox"][1]

        base = 50
        xmin = xmin - base if (xmin - base) >= 0 else 0
        ymin = ymin - base if (ymin - base) >= 0 else 0
        xmax = xmax + base if (xmax + base) <= w else w
        ymax = ymax + base if (ymax + base) <= h else h

        sub_image = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
        mask = self.coco.annToMask(ann)
        mask = mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        # sub_image = Image.fromarray(sub_image)
        # mask = Image.fromarray(mask)
        sub_image = resize(sub_image, (128, 128))
        mask = resize(mask*255, (128, 128))
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        # mask *= 255
        # sub_image = sub_image.astype(np.uint8)
        # mask = mask.astype(np.uint8)
        return sub_image, mask
        
    def save2npy(self):
        print("saving images into .npy")
        Imgs = []
        Masks = []
        for image in tqdm(self.images):
            imagePath = ROOT + image['imagePath'][1:]
            img = imageio.imread(imagePath)
            id = image['id']
            anns_ids = self.coco.getAnnIds(imgIds = [id])
            anns = self.coco.loadAnns(ids=anns_ids)
            for ann in anns:
                sub_image, mask = self.get_sub(img, ann)
                Imgs.append(sub_image)
                Masks.append(mask)
            os.makedirs('./temp', exist_ok=True)
        with open('./temp/images.npy', 'wb') as f:
            np.save(f, np.array(Imgs))
        with open('./temp/masks.npy', 'wb') as f:
            np.save(f, np.array(Masks))

        print("save finished")
            
    def load_npy(self):
        with open('./temp/images.npy', 'rb') as f:
            train_x = np.load(f)
        with open('./temp/masks.npy', 'rb') as f:
            train_y = np.load(f)
        return train_x, train_y
        
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
        
        
# class Prediction(torch.utils.data.Dataset):
#     def __init__(self, img_dir, size=256, transforms=transforms.ToTensor()):
#         super().__init__()
#         self.img_dir = img_dir
#         self.size = size
#         self.img_list = os.listdir(f'{img_dir}')
#         self.transforms = transforms
        
#     def readImage(self, img_id):
#         # img = Image.open(img_id)
#         img = imageio.imread(img_id)
#         if img_id.endswith('tiff'):
#             img = img[:, :, :3]
#         img = Image.fromarray(img)
#         transform = transforms.Resize((self.size, self.size))
#         return transform(img)
    
#     def __len__(self) -> int:
        
#         return len(self.img_list)
    
#     def __getitem__(self, index):
        
#         img_id = self.img_list[index]
        
#         img = self.readImage(f'{self.img_dir}/{img_id}')

#         if self.transforms:
#             img = self.transforms(img)
#         if img_id.endswith('tiff'):
#             img_id.replace('tiff', 'jpg')
#         return img, img_id