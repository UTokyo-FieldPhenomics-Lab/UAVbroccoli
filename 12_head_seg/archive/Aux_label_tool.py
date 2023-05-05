import os
import random

import json
import numpy as np
import pandas as pd
from PIL import Image
import imageio


import torch

from torchvision import transforms


from bisenet.bisenetv2 import BiSeNetV2
from utils.mask2polygon import mask2polygon
from utils.tolabelme import write_json

from shutil import copyfile

ROOT = 'G:/Shared drives/broccoliProject/'
MAP_PATH = ROOT + '13_roi_on_raw/'
IMG_PATH = MAP_PATH
INDEX_PATH = ROOT + '13_roi_on_raw/json_index.csv'
JSON_PATH = ROOT + '13_roi_on_raw/train/'

# ROOT_ON_RAW_PATH = 'G:/Shared drives/broccoliProject/11_labelme_json/root_on_raw.json/'

# BACK_UP = 'G:/Shared drives/broccoliProject/11_labelme_json/aux_iter_backup'
# MASK_PATH = './deeplab/test/masks/'
# os.makedirs(MASK_PATH, exist_ok=True)

device = 'cuda' if torch.cuda.is_available==True else 'cpu'

model = BiSeNetV2(n_classes=2)
model.to(device)

model_weight = torch.load('./bisenet/checkpoints/best_model.tar')
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
        pred, *logits_aux = model(batch)

        masks = pred.permute(0, 2, 3, 1).detach().cpu().numpy()
        # images = batch.permute(0, 2, 3, 1).detach().cpu().numpy()
        # print(masks[0])
        # masks[masks >= 0.5] = 10
        # masks[masks < 0.5] = 255
        
        masks = masks.argmax(3).reshape(-1, 128, 128, 1)
        
        return np.array(masks, dtype=np.uint8)


def one_image(imageName, map_label, base=70, root=IMG_PATH):
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
        mask[mask > 50] = 255
        mask[mask <= 50] = 0
        

        mask = np.asarray(mask).transpose((1,2,0))
        
        # print(mask.shape)
        one_mask[y0:y1, x0:x1, :] = mask
    # imageio.imsave('./test.png', one_mask)
    return mask, mask2polygon(one_mask[:, :, 0])
    
    # imageio.imwrite(os.path.join(MASK_PATH, name), mixed)
    
def Aux_label(item, version='v1'):
    """[summary]

    Args:
        dir_path ([string]): [path to target image folder]

    Returns:
        [list]: [list of cropped images with shape (n, c, w, h)]
    """    

    date, img_name, r_path, status, _ = item
    # print(img_path)
    project_name = r_path.split('/')[-2]
    map_label = f'{MAP_PATH}{project_name}.json'
    # print(map_label)
    print(f"start processing on: {img_name}")
    _, polygons = one_image(img_name, map_label=map_label)
    print("Prediction finished")

    # write_json
    labelme = {
        'version': "4.5.7",
        'flags': {},
        'shapes': [],
        'imagePath': None,
        'imageData': None,
        'imageHeight': None,
        'imageWidth': None
    }

    labelme['imagePath'] = '.' + r_path

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
    json_name = f'aux_{version}_{date}_{temp}.json'
    print(f"saving result to {json_name}")
    with open(os.path.join(JSON_PATH, json_name), 'w+') as f:
        f.write(json.dumps(labelme, indent=1))

    copyfile(os.path.join(JSON_PATH, json_name), f'{JSON_PATH}/../aux_backup/{json_name}')
    print("result saved")

def Random_select_aux(number, version='v1', status='aux', save=True):
    df = pd.read_csv(INDEX_PATH, index_col=None)
    filtered = df[df['status']=='blank']
    grouped = filtered.groupby('date')
    target_list = [group.sample(n=number).date_filename.item() for _, group in grouped]
    target_df = df[df['date_filename'].isin(target_list)]
    for idx, item in target_df.iterrows():
        Aux_label(item, version)
    if save:
        df.loc[df['date_filename'].isin(target_list), ['status']] = status
        df.to_csv(INDEX_PATH, index=False)

def target_select_aux(target_list, version='v0', status='verify', save=True):
    df = pd.read_csv(INDEX_PATH, index_col=None)
    # df['date_filename'] = df['date'].map(str) + '_' + df['file_name']
    target_df = df[df['date_filename'].isin(target_list)]
    for idx, item in target_df.iterrows():
        Aux_label(item, version)
    if save:
        df.loc[df['date_filename'].isin(target_list), ['status']] = status
        df.to_csv(INDEX_PATH, index=False)
if __name__ == "__main__":
    # pass
    # Random_select_aux(6)
    # target_list = ['20200518_60_DJI_0462.JPG',
    #                 '20200520_66_DJI_0994.JPG',
    #                 '20200520_66_DJI_0995.JPG',
    #                 '20200520_66_DJI_0996.JPG',
    #                 '20200520_89_DJI_0993.JPG',
    #                 '20200520_135_DJI_0924.JPG',
    #                 '20200522_34_DJI_0778.JPG',
    #                 '20200522_34_DJI_0812.JPG',
    #                 '20200522_34_DJI_0813.JPG',
    #                 '20200522_90_DJI_0724.JPG',
    #                 '20200522_162_DJI_0625.JPG',
    #                 '20200525_50_DJI_0368.JPG',
    #                 '20200525_50_DJI_0326.JPG',
    #                 '20200525_50_DJI_0369.JPG',
    #                 '20200525_79_DJI_0336.JPG',
    #                 '20200525_86_DJI_0254.JPG',
    #                 '20200526_274_DJI_0743.JPG',
    #                 '20200526_274_DJI_0742.JPG',
    #                 '20200526_274_DJI_0745.JPG',
    #                 '20200526_327_DJI_0664.JPG',
    #                 '20200526_5_DJI_0071.JPG',
    #                 '20200528_240_DJI_0377.JPG',
    #                 '20200528_240_DJI_0338.JPG',
    #                 '20200528_240_DJI_0378.JPG',
    #                 '20200528_269_DJI_0302.JPG',
    #                 '20200528_1_DJI_0651.JPG']

    # target_list = ['20200528_45_DJI_0589.JPG']
    # target_select_aux(target_list, version='v0', status='verify', save=False)

    # Random_select_aux(1, version=0, status='train')

    target_list = ['20200518_3_DJI_0489.JPG',
                    '20200520_34_DJI_0048.JPG',
                    '20200522_33_DJI_0811.JPG',
                    '20200525_36_DJI_0398.JPG',
                    '20200526_44_DJI_0062.JPG',
                    '20200528_31_DJI_0632.JPG']

    target_select_aux(target_list, version='v0', status='train', save=True)