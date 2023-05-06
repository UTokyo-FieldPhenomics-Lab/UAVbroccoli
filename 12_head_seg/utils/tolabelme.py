import os
import json
import imageio
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = 'G:/Shared drives/broccoliProject/'
PROJECT_PATH = ROOT + '/11_labelme_json/root_on_raw.json/'
JSON_PATH = ROOT + '/11_labelme_json/json'
# os.makedirs(JSON_PATH, exist_ok=True)


def write_json_old(img_path, prefix):
    
    labelme = {
        'version': "4.5.7",
        'flags': {},
        'shapes': [],
        'imagePath': None,
        'imageData': None,
        'imageHeight': None,
        'imageWidth': None
    }
    
    # print(img_path)
    labelme['imagePath'] = '../../' + img_path[33:]
    
    json_name = prefix + '_' + img_path.split('\\')[-1].split('.')[-2] + '.json'
    
    # print(json_name)
    with open(os.path.join(JSON_PATH, json_name), 'w+') as f:
        f.write(json.dumps(labelme, indent=1))

def save_lbme_json(img_path, polygons, json_save_path, img_w=None, img_h=None):
    labelme = {
        'version': "5.0.1",
        'flags': {},
        'shapes': [],
        'imagePath': None,
        'imageData': None,
        'imageHeight': img_h,
        'imageWidth': img_w
    }
    labelme['imagePath'] = img_path

    for polygon in polygons:  
        coords = {
            "label": "broccoli",
            "points": polygon[:-1],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme['shapes'].append(coords)

    with open(json_save_path, 'w+') as f:
        f.write(json.dumps(labelme, indent=1))

if __name__ == "__main__":
    # jsonsFile= [entry for entry in os.scandir(PROJECT_PATH) if entry.name.endswith('.json')]
    # # print(imagesPath[0])
    # for json_file in jsonsFile:
    #     with open(json_file, 'r', encoding='utf-8') as f:
    #         label_data = json.load(f)
    #     prefix = json_file.name.split('_')[3]
        
    #     for img_name in tqdm(label_data.keys(), total = len(label_data.keys())):
    #         imagePath = ROOT + label_data[img_name]['imagePath'][40:]
    #         write_json(imagePath, prefix)
    pass
        