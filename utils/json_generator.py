import os
import json
import imageio
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_PATH = 'I:/Shared drives/broccoliProject/00_rgb_raw'
JSON_PATH = 'I:/Shared drives/broccoliProject/11_labelme_json/blank'
os.makedirs(JSON_PATH, exist_ok=True)
DIR_LIST = ['broccoli_tanashi_5_20200518_P4M_10m',
            'broccoli_tanashi_5_20200520_P4M_10m',
            'broccoli_tanashi_5_20200522_P4M_10m_after',
            'broccoli_tanashi_5_20200525_P4M_10m',
            'broccoli_tanashi_5_20200526_P4M_10m_after',
            'broccoli_tanashi_5_20200528_P4M_10m_after']

def write_json(img_path, prefix):
    labelme = {
        'version': "4.5.7",
        'flags': {},
        'shapes': [],
        'imagePath': None,
        'imageData': None,
        'imageHeight': None,
        'imageWidth': None
    }
    
    img = imageio.imread(img_path)
    h, w, _ = img.shape
    labelme['imageHeight'] = h
    labelme['imageWidth'] = w
    labelme['imagePath'] = img_path
    
    json_name = 'blank_' + prefix + '_' + img_path.split('/')[-1].split('.')[-2] + '.json'
    
    # print(json_name)
    with open(os.path.join(JSON_PATH, json_name), 'w+') as f:
        f.write(json.dumps(labelme))
    
   
def from_dir(dir, prefix):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
    imagesPath= [f'{dir}/{entry.name}' for entry in os.scandir(dir) if entry.name.split('.')[-1].lower() in img_formats]
    # print(imagesPath[0])
    for imagePath in tqdm(imagesPath):
        write_json(imagePath, prefix)
        
if __name__ == "__main__":
    for DIR in DIR_LIST:
        PATH = f'{PROJECT_PATH}/{DIR}'
        prefix = DIR.split('_')[-4]
        from_dir(PATH, prefix)