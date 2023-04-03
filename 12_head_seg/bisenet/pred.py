import os
import torch
import re
import numpy as np
import json
import pathlib
import sys
sys.path.append("..")
from .. import easyidp as idp

from tqdm import tqdm

import imageio as iio
from skimage.draw import polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2

from bisenet.bisenetv2 import BiSeNetV2

from utils.mask2polygon import mask2polygon

import config

def predict_batch(model, batch_images, device):
    """[summary]

    Args:
        model ([model]): [deeplab model]
        img ([list]): [batch of images]

    Returns:
        [type]: [description]
    """    
    model.eval()
    with torch.no_grad():
        
        batch_images.to(device)

        pred, *_ = model(batch_images)

        masks = pred.permute(0, 2, 3, 1).detach().cpu().numpy()
        masks = masks.argmax(3).reshape(-1, 128, 128, 1)
        
        return np.array(masks, dtype=np.uint8)
    

def pred_whole_image(img_path, coords, model, base=75, transform=None):
    img = iio.imread(img_path)
    h, w, _ = img.shape
    whole_mask = np.zeros((h, w, 1), dtype=np.uint8)
    
    # convert to numpy array
    coords = np.array(coords, dtype=np.int32)

    # calcualte x0, y0, x1, y1
    y0 = coords[:, 1] - base
    x0 = coords[:, 0] - base
    y1 = coords[:, 1] + base
    x1 = coords[:, 0] + base

    # avoid minus number
    y0[y0 < 0] = 0
    x0[x0 < 0] = 0
    y1[y1 > h] = h
    x1[x1 > w] = w
    
    bboxs = np.array(list(zip(y0, x0, y1, x1)))

    if transform == None:
        transform = A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

    # prepare batch images ([n, c, w, h])
    croped_images_batch = []
    for y_0, x_0, y_1, x_1 in bboxs:
        croped_img = img[y_0:y_1, x_0:x_1, :3]
        croped_img = transform(image=croped_img)['image']
        croped_images_batch.append(croped_img)

    croped_images_batch = torch.stack(croped_images_batch)

    # prediction
    masks = predict_batch(model, croped_images_batch)

    # post process
    for box, mask in zip(bboxs, masks):
        # print(np.unique(mask))
        y0, x0, y1, x1 = box
        # print(mask, mask.shape) 
        mask = transform.Resize((y1-y0, x1-x0))(torch.tensor(mask).permute((2,0,1)))*255
        mask[mask > 50] = 255
        mask[mask <= 50] = 100

        mask = np.asarray(mask).transpose((1,2,0))
        
        # print(mask.shape)
        whole_mask[y0:y1, x0:x1, :] = mask
    # iio.imsave('./test.png', whole_mask)
    return mask2polygon(whole_mask[:, :, 0])

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

if __name__ == "__main__":
    run_images = config.img_dirs
    device = 'cuda' if torch.cuda.is_available==True else 'cpu'

    # load model
    model = BiSeNetV2(n_classes=2).to(device)
    model_weight = torch.load(config.model_weight)
    model_weight = model_weight["model_state_dict"]
    model.load_state_dict(model_weight, strict=False)

    # print(test)
    for img_dir in run_images:
        path = pathlib.Path(config.img_dir)
        project_name = path.stem
        json_path = path.parent / (path.name + '.json')
        if not json_path.exists():
            raise FileNotFoundError(f'Json file {json_path} does not exist.')
        
        info_map = idp.jsonfile.read_json(json_path)
        for croped_image_name, values in info_map.items():
            croped_image_path = values['cropedImagePath']
            coords = values['headCoordOnCroppedImage']

            polygons = pred_whole_image(croped_image_path, coords, model, base=75)
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
            # date = img_path.split('/')[-2].split('_')[3]
            labelme['imagePath'] = croped_image_path

            for polygon in polygons:  
                coords = {
                    "label": "broccoli",
                    "points": polygon,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                labelme['shapes'].append(coords)

            result_json_name = f'pred_{project_name}_{croped_image_name}.json'
            print(f"saving result to {result_json_name}")

            with open(os.path.join(config.out_dir, result_json_name), 'w+') as f:
                f.write(json.dumps(labelme, indent=1))
            print("result saved")
