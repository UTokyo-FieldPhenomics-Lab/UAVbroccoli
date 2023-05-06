import os
import re
import numpy as np
from PIL import Image
from sklearn import model_selection

from pycocotools.coco import COCO
from tqdm import tqdm

from shutil import copyfile

ROOT = 'G:/Shared drives/broccoliProject2021/'
JSON_PATH = ROOT + '11_instance_seg/coco_json/yolo_21.json'
OUTPUT_PATH = 'C:/Users/ddgip/works/UAVbroccoli/broccoli_data_2021'

def process_data(images, data_type="train"):
    for im in tqdm(images, total=len(images)):

        annIds = coco.getAnnIds(imgIds=[im['id']])
        anns = coco.loadAnns(annIds)
        
        image_name = im['file_name']
        src = ROOT + im['imagePath'][6:]
        dst = f'{OUTPUT_PATH}/images/{data_type}'
        os.makedirs(dst, exist_ok=True)
        copyfile(src, f'{dst}/{image_name}')
        filename = re.sub('jpg', 'txt', image_name, flags=re.IGNORECASE)
        width = im['width']
        height = im['height']
        yolo_data = []
        for i in range(len(anns)):
            cat = anns[i]["category_id"] - 1
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
            ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

            x = (xmin + xmax)/2
            y = (ymin + ymax)/2

            w = xmax - xmin
            h = ymax-ymin

            x /= width
            w /= width
            y /= height
            h /= height
            
            yolo_data.append([cat, x, y, w, h])
            
        yolo_data = np.array(yolo_data)
        path = f'{OUTPUT_PATH}/labels/{data_type}'
        os.makedirs(path, exist_ok=True)
        # save labels
        np.savetxt(
            f"{path}/{filename}",
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        
        
        
if __name__ == "__main__":
    coco = COCO(JSON_PATH)
    print(coco.info)
    
    # # get all category names
    # cats = coco.loadCats(coco.getCatIds())
    # cats = [cat['name'] for cat in cats]
    
    # get all ImgIds and images
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    
    #train, validation split
    # train, valid = model_selection.train_test_split(
    #     images,
    #     test_size=0.1,
    #     random_state=42,
    #     shuffle=True        
    # )
    
    process_data(images, data_type="train")
    process_data(images, data_type="validation")
    
