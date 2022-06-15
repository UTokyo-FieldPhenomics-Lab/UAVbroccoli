import os
import torch
import numpy as np

import json

from utils.labelme2coco import labelme2json
from pycocotools.coco import COCO
from tqdm import tqdm

import imageio
from PIL import Image
from torchvision import transforms
from bisenet.bisenetv2 import BiSeNetV2

ROOT = 'G:/Shared drives/broccoliProject/'
VALID_PATH = ROOT + '13_roi_on_raw/valid'
MAP_PATH = ROOT + '13_roi_on_raw/'

device = 'cuda' if torch.cuda.is_available==True else 'cpu'

model = BiSeNetV2(n_classes=2).to(device)


model_weight = torch.load('./bisenet/checkpoints/best_model.tar')
model_weight = model_weight["model_state_dict"]

model.load_state_dict(model_weight, strict=False)
    
def iou_mean(pred, target, n_classes=1):

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

def one_image(imageName, map_label, base=70, root=ROOT):
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

def validation():
    json_list = [entry for entry in os.scandir(VALID_PATH) if entry.name.endswith('.json')]
    print(f'{len(json_list)} validation files:')
    print(entry.name for entry in json_list)
    date_list = ['20200518', '20200520', '20200522', '20200525', '20200526', '20200528']
    ious = []
    for date in date_list:
        name_list = [entry.name for entry in json_list if entry.name.split('_')[2] == date]
        coco_label = labelme2json(VALID_PATH, name_list)
        os.makedirs('./bisenet/temp', exist_ok=True)
        with open('./bisenet/temp/validation.json', 'w+') as f:
            f.write(json.dumps(coco_label))
        
        coco = COCO('./bisenet/temp/validation.json')

        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
        
        valid_iou = 0

        for image in tqdm(images):
            img_path = ROOT + image['imagePath'][3:]
            
            img_name = image['imagePath'].split('\\')[-1]
            project_name = image['imagePath'].split('\\')[-2]
            # print(img_path)
            map_label = f'{MAP_PATH}{project_name}.json'
            root = f'{MAP_PATH}{project_name}/'
            pred_mask = one_image(img_name, map_label=map_label, root=MAP_PATH)

            id = image['id']
            anns_ids = coco.getAnnIds(imgIds = [id])
            anns = coco.loadAnns(ids=anns_ids)
            mask = coco.annToMask(anns[0])
            # print(len(anns))
            for i in range(len(anns)):
                mask = mask | coco.annToMask(anns[i])

            miou = iou_mean(pred_mask, mask)
            # print(miou)
            valid_iou += miou
        ious.append(valid_iou / len(images))
    average_iou = sum(ious)/6
    # imageio.imsave('./mask.png', mask*255)
    # imageio.imsave('./pred_mask.png', pred_mask*255)
    print(f'iou on each date: {ious}')
    print(f'average iou on validation set: {average_iou:.3f}')


if __name__ == "__main__":
    validation()