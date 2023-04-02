import os
import torch
import re
import numpy as np
import pandas as pd
import json

import sys
sys.path.append("..") 
from utils.labelme2coco import labelme2json
from pycocotools.coco import COCO
from tqdm import tqdm

import imageio
from skimage.draw import polygon
from PIL import Image
from torchvision import transforms
from bisenet.bisenetv2 import BiSeNetV2

ROOT = 'G:/Shared drives/broccoliProject/'
# ROOT = 'G:/Shared drives/broccoliProject2021/'
VALID_PATH = ROOT + '13_roi_on_raw/valid'
MAP_PATH = ROOT + '13_roi_on_raw/'

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

def one_image(imageName, map_label, base=75, root=ROOT):
    transform1 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # transform2 = 
    
    with open(map_label, 'r', encoding='utf-8') as f:
        # lines = f.readlines()
        label_data = json.load(f)
        
    imagePath = label_data[imageName]['imagePath']
    imagePath = root + imagePath
    
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
        mask[mask <= 50] = 0
        mask[mask > 50] = 1
        
        mask = np.asarray(mask).transpose((1,2,0))
        
        # print(mask.shape)
        one_mask[y0:y1, x0:x1, :] = mask
    # imageio.imsave('./test.png', one_mask)
    return one_mask

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

def validation():
    json_list = [entry for entry in os.scandir(VALID_PATH) if entry.name.endswith('.json')]
    print(f'{len(json_list)} validation files:')
    print(entry.name for entry in json_list)
    # date_list = ['20210512', '20210514', '20210515', '20210519', '20210520', '20210526']
    date_list = ['20200518', '20200520', '20200522', '20200525', '20200526', '20200528']
    ious = []
    med_ious = []
    for date in date_list:
        name_entry = [entry for entry in json_list if entry.name.split('_')[2] == date]
        coco_label = labelme2json(name_entry)
        os.makedirs('./temp', exist_ok=True)
        with open('./temp/validation.json', 'w+') as f:
            f.write(json.dumps(coco_label))
        
        coco = COCO('./temp/validation.json')

        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
        
        valid_iou = 0
        valid_med_iou = 0
        for image in tqdm(images):
            img_path = ROOT + image['imagePath'][3:]
            
            img_name = image['imagePath'].split('\\')[-1]
            project_name = image['imagePath'].split('\\')[-2]
            # print(img_path)
            map_label = f'{MAP_PATH}{project_name}.json'
            roi_label = f'{MAP_PATH}{project_name}.csv'
            root = f'{MAP_PATH}{project_name}/'
            roi_df = pd.read_csv(roi_label, index_col=None)
            roi_df['index'] = roi_df['id'].map(str) + '_' + roi_df['image']
            roi_row = roi_df[roi_df['index']==img_name]
            np_roi = get_roi(roi_row)

            roi_mask = polygon2mask(np_roi, shape=(1500, 1500))

            pred_mask = one_image(img_name, map_label=map_label, root=MAP_PATH)

            id = image['id']
            anns_ids = coco.getAnnIds(imgIds = [id])
            anns = coco.loadAnns(ids=anns_ids)
            mask = coco.annToMask(anns[0])
            # print(len(anns))
            for i in range(len(anns)):
                mask = mask | coco.annToMask(anns[i])

            miou = iou_mean(pred_mask, mask)
            # print(pred_mask.shape, mask.shape)
            med_miou = iou_mean(np.multiply(roi_mask, pred_mask[..., 0]), np.multiply(roi_mask, mask))
            # print(miou)
            # print(med_miou)
            valid_iou += miou
            valid_med_iou += med_miou

        ious.append(round(valid_iou / len(images), 5))
        med_ious.append(round(valid_med_iou / len(images), 5))
    #average_iou
    ious.append(round(sum(ious)/6, 5))
    #average_med_iou
    med_ious.append(round(sum(med_ious)/6, 5))

    print(*date_list, sep='\t')
    print(*ious, sep='\t')
    print(*med_ious, sep='\t')


    # print(f'average iou on validation set: {average_iou:.5f}')
    # print(f'med_iou on each date: {med_ious}')
    # print(f'average med_iou on validation set: {average_med_iou:.5f}')


if __name__ == "__main__":
    validation()