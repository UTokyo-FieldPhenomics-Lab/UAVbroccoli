import os
import json
import datetime as dt
from shapely.geometry import Polygon
import numpy as np
from pathlib import Path

def at_path(path):
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            cur_path = os.getcwd()
            os.chdir(path)

            rst = func(*args, **kwargs)

            os.chdir(cur_path)
            return rst
        return wrapper
    return decorator

def labelme2coco(json_entry):
    
    # setup COCO dataset container and info
    coco = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }

    for json_file in json_entry:
        with open(json_file.path, 'r', encoding='utf-8') as f:
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
        p = Path(label_data['imagePath'])
        image_name = p.name

        image = {
            "id": image_name,
            "width": label_data['imageWidth'],
            "height": label_data['imageHeight'],
            "file_name": image_name,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
            "imagePath": p.resolve().__str__()
        }

        coco['images'].append(image)
        
        # convert Labelbox Polygon to COCO Polygon format
        for object_ in label_data['shapes']:
            # for polygon type
            if object_['shape_type'] == 'polygon':
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

            
            if object_['shape_type'] == 'rectangle':
                rectangle = object_['points']
                rectangle_ = np.array(rectangle)
                x0, y0 = rectangle_[0, :]
                x1, y1 = rectangle_[1, :]
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
                
                annotation = {
                    "id": len(coco['annotations']) + 1,
                    "image_id": image_name,
                    "category_id": cat_id,
                    "segmentation": [],
                    "area": (x1-x0)*(y1-y0),  # float
                    "bbox": [x0, y0, x1-x0, y1-y0],
                    "iscrowd": 0
                }

                coco['annotations'].append(annotation)
        # print(coco)
    return coco