import os
import torch
import re
import numpy as np
import pandas as pd
import json

import sys
sys.path.append("..")
from tqdm import tqdm

import imageio
from skimage.draw import polygon
from PIL import Image
from torchvision import transforms
from bisenet.bisenetv2 import BiSeNetV2

from utils.mask2polygon import mask2polygon

ROOT = 'G:/Shared drives/broccoliProject/'
MAP_PATH = ROOT + '13_roi_on_raw/'
IMG_PATH = MAP_PATH
INDEX_PATH = ROOT + '13_roi_on_raw/json_index.csv'
JSON_PATH = ROOT + '13_roi_on_raw/pred/'
os.makedirs(JSON_PATH, exist_ok=True)
device = 'cuda' if torch.cuda.is_available==True else 'cpu'

model = BiSeNetV2(n_classes=2).to(device)


model_weight = torch.load('./checkpoints/v5/epoch200.tar')
# model_weight = torch.load('./ckpt/epoch600.tar')
model_weight = model_weight["model_state_dict"]

model.load_state_dict(model_weight, strict=False)
    
def iou_mean(pred, target, roi=None, n_classes=1):

    ious = []
    iousSum = 0

    # print(pred.shape, target.shape)
    pred = torch.from_numpy(pred)
    pred = pred.reshape(-1)
    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.reshape(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        # print(intersection)
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        # print(union)
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))

    return iousSum/n_classes

def predict_batch(model, batch):
    """[summary]

    Args:
        model ([model]): [deeplab model]
        img ([list]): [batch of images]

    Returns:
        [type]: [description]
    """    
    model.eval()
    with torch.no_grad():
        
        batch.to(device)
        # print(img)
        pred, *logits_aux = model(batch)
        # print(pred.shape)
        masks = pred.permute(0, 2, 3, 1).detach().cpu().numpy()
        # images = batch.permute(0, 2, 3, 1).detach().cpu().numpy()
        # print(masks[0])
        # masks[masks >= 0.5] = 10
        # masks[masks < 0.5] = 255
        
        masks = masks.argmax(3).reshape(-1, 128, 128, 1)
        
        return np.array(masks, dtype=np.uint8)

def one_image(imagePath, map_label, base=75, root=ROOT):
    imagePath = MAP_PATH + imagePath
    with open(map_label, 'r', encoding='utf-8') as f:
        # lines = f.readlines()
        label_data = json.load(f)

    transform1 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    imageName = imagePath.split('/')[-1]
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
        mask[mask <= 50] = 100
        # print('masks:', np.unique(mask))
        mask = np.asarray(mask).transpose((1,2,0))
        
        # print(mask.shape)
        one_mask[y0:y1, x0:x1, :] = mask
    # print('test')
    imageio.imsave('./test.png', one_mask)
    return mask2polygon(one_mask[:, :, 0])

def get_roi(target):
    roi_str = target['roi'].values[0]
    off_x = target['offset_x'].values[0]
    off_y = target['offset_y'].values[0]

    roi_str = roi_str.replace("\n", ',').replace('[ ', '[').replace(' [', '[')
    roi_str = re.sub("\s+", ',', roi_str)

    np_roi = np.asarray(eval(roi_str))
    np_roi[:,0] = np_roi[:,0] - off_x
    np_roi[:,1] = np_roi[:,1] - off_y
    return np_roi

def polygon2mask(poly, shape):
    a_mask = np.zeros(shape=(1500, 1500), dtype="bool") # original
    rr, cc = polygon(poly[:,0], poly[:,1], shape)
    a_mask[cc,rr] = True   
    return a_mask

def read_batch(csv_path):
    with open(csv_path, newline='') as f:
        data = [i.replace(",\r\n", "").replace("Y:\\hwang_Pro\\data\\2020_tanashi_broccoli/13_roi_on_raw/", "") for i in f.readlines()]
    return data


if __name__ == "__main__":
    run_images = read_batch('G:/Shared drives/broccoliProject/13_roi_on_raw/run_images18.csv')
    # print(test)
    for img_path in tqdm(run_images):
        project_name = img_path.split('/')[-2]
        
        map_label = f'{MAP_PATH}{project_name}.json'
        # print(project_name, map_label)
        polygons = one_image(img_path, map_label=map_label, base=75)
        # print(polygons)
        # write_json
        labelme = {
            'version': None,
            'flags': {},
            'shapes': [],
            'imagePath': None,
            'imageData': None,
            'imageHeight': None,
            'imageWidth': None
        }
        img_name = img_path.split('/')[-1]
        date = img_path.split('/')[-2].split('_')[3]
        labelme['imagePath'] = '../' + img_path

        for polygon in polygons:  
            coords = {
                "label": "broccoli",
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme['shapes'].append(coords)
        temp = img_name.split('.')[0]
        json_name = f'pred_{date}_{temp}.json'
        print(f"saving result to {json_name}")
        with open(os.path.join(JSON_PATH, json_name), 'w+') as f:
            f.write(json.dumps(labelme, indent=1))
        print("result saved")