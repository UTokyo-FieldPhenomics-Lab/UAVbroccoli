import os
import requests
import numpy as np
from PIL import Image
from sklearn import model_selection

from pycocotools.coco import COCO
from tqdm import tqdm


JSON_PATH = './data/annotation/coco_1_26.json'
OUTPUT_PATH = './data/broccoli_data'


def process_data(images, data_type="train"):
    for im in tqdm(images, total=len(images)):
        try:
            response = requests.get(im['coco_url'], stream=True)
        except requests.exceptions.MissingSchema as e:
            logging.exception(('"coco_url" field must be a URL. '))
            continue
        except requests.exceptions.ConnectionError as e:
            logging.exception(f"Failed to fetch image from {im['coco_url']}")
            continue

        response.raw.decode_content = True
        img = Image.open(response.raw)
        
        annIds = coco.getAnnIds(imgIds=[im['id']])
        anns = coco.loadAnns(annIds)
        
        image_name = im['file_name']
        filename = image_name.replace(".jpg", ".txt")
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
        
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # save labels
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{filename}"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        
        #save iamges
        img.save(os.path.join(OUTPUT_PATH, f'images/{data_type}/{image_name}'))
        
        
if __name__ == "__main__":
    coco = COCO(JSON_PATH)
    print(coco.info)
    
    # # get all category names
    # cats = coco.loadCats(coco.getCatIds())
    # cats = [cat['name'] for cat in cats]
    
    # get all ImgIds and images
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    
    # train, validation split
    train, valid = model_selection.train_test_split(
        images,
        test_size=0.1,
        random_state=42,
        shuffle=True        
    )
    
    process_data(train, data_type="train")
    process_data(valid, data_type="validation")