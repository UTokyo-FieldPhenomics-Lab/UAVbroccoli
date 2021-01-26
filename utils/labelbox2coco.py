import os
import json
import datetime as dt
import logging
from shapely.geometry import Polygon
import requests
from PIL import Image
from tqdm import tqdm


def from_json(labeled_data, coco_output, im_output):
    # read labelbox JSON output
    with open(labeled_data, 'r', encoding='utf-8') as f:
        # lines = f.readlines()
        label_data = json.load(f)

    # setup COCO dataset container and info
    coco = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }

    coco['info'] = {
        'year': dt.datetime.now(dt.timezone.utc).year,
        'version': None,
        'description': label_data[0]['Project Name'],
        'contributor': label_data[0]['Created By'],
        'url': 'labelbox.com',
        'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
    }

    for data in tqdm(label_data, total=len(label_data)):
        # skip images without label
        if data['Label'] == {}:
            continue
        
        # Download and get image name
        try:
            response = requests.get(data['Labeled Data'], stream=True)
        except requests.exceptions.MissingSchema as e:
            logging.exception(('"Labeled Data" field must be a URL. '
                              'Support for local files coming soon'))
            continue
        except requests.exceptions.ConnectionError as e:
            logging.exception('Failed to fetch image from {}'
                              .format(data['Labeled Data']))
            continue

        response.raw.decode_content = True
        im = Image.open(response.raw)
        width, height = im.size
        
        image = {
            "id": data['ID'],
            "width": width,
            "height": height,
            "file_name": data['External ID'],
            "license": None,
            "flickr_url": data['Labeled Data'],
            "coco_url": data['Labeled Data'],
            "date_captured": None,
        }

        coco['images'].append(image)
        
        # save image to data/train/
        # os.makedirs(im_output, exist_ok=True)
        # im.save(os.path.join(im_output, data['External ID']))

        # convert Labelbox Polygon to COCO Polygon format
        for object_ in data['Label']['objects']:
            polygon = object_['polygon']
            # add categories
            cat = object_['value']
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
            segmentation = [list(i.values()) for i in polygon]
            m = Polygon(segmentation)
            
            annotation = {
                "id": len(coco['annotations']) + 1,
                "image_id": data['ID'],
                "category_id": cat_id,
                "segmentation": [segmentation],
                "area": m.area,  # float
                "bbox": [m.bounds[0], m.bounds[1],
                            m.bounds[2]-m.bounds[0],
                            m.bounds[3]-m.bounds[1]],
                "iscrowd": 0
            }

            coco['annotations'].append(annotation)

    with open(coco_output, 'w+') as f:
        f.write(json.dumps(coco))

if __name__ == "__main__":
    json_file = './data/annotation/labelbox_1_26.json'
    im_output = './data/train'
    coco_output = './data/annotation/coco_1_26.json'
    from_json(json_file, coco_output, im_output)