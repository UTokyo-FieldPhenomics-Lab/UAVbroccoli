import os
import json
import datetime as dt
from shapely.geometry import Polygon
import requests
from tqdm import tqdm
import numpy as np

json_path = 'I:/Shared drives/broccoliProject/11_labelme_json/json'
def labelme2json(json_list):
    
    # setup COCO dataset container and info
    coco = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }

    for json_file in json_list:
            # read labelbox JSON output
        with open(f'{json_path}/{json_file}', 'r', encoding='utf-8') as f:
            # lines = f.readlines()
            label_data = json.load(f)
        
        coco['info'] = {
            'year': dt.datetime.now(dt.timezone.utc).year,
            'version': label_data['version'],
            'flags': label_data['flags'],
            # 'contributor': label_data[0]['Created By'],
            # 'url': 'labelbox.com',
            'date_created': dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        part1 = label_data['imagePath'].split('\\')[-2].split('_')[3]
        part2 = label_data['imagePath'].split('\\')[-1]
        image_name = part1 + '_'+ part2
        image = {
            "id": image_name,
            "width": label_data['imageWidth'],
            "height": label_data['imageHeight'],
            "file_name": image_name,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
            "imagePath": label_data['imagePath']
        }

        coco['images'].append(image)
        
        # convert Labelbox Polygon to COCO Polygon format
        for object_ in label_data['shapes']:
            polygon = object_['points']
            polygon_ = np.array(polygon)
            # print('polygon:')
            # print(np.min(polygon_[:, 0]), np.max(polygon_[:, 0]), np.max(polygon_[:, 1]), np.min(polygon_[:, 1]))
            # add categories
            cat = object_['label']
            try:
                # check if label category exists in 'categories' field
                cat_id = [c['id'] for c in coco['categories']
                        if c['supercategory'] == cat][0]
            except IndexError as e:
                cat_id = len(coco['categories']) + 1
                category = {
                    'supercategory': cat,
                    'id': len(coco['categories']) + 1,
                    'name': cat
                }
                coco['categories'].append(category)
                
            # add polygons
            segmentation = [j for sub in polygon for j in sub]
            m = Polygon(polygon)
            
            annotation = {
                "id": len(coco['annotations']) + 1,
                "image_id": image_name,
                "category_id": cat_id,
                "segmentation": [segmentation],
                "area": m.area,  # float
                "bbox": [m.bounds[0], m.bounds[1],
                            m.bounds[2]-m.bounds[0],
                            m.bounds[3]-m.bounds[1]],
                "iscrowd": 0
            }

            coco['annotations'].append(annotation)
    return coco

if __name__ == "__main__":
    json_dir = 'I:/Shared drives/broccoliProject/11_labelme_json/test'
    # im_output = './data/train'
    coco_output = './temp.json'
    coco = labelme2json(json_dir)
    with open(coco_output, 'w+') as f:
        f.write(json.dumps(coco))