from config import *

import os
import json
import numpy as np
from pathlib import Path
from shutil import copyfile

# create yolo database
if not os.path.exists(train_database_path):
    os.makedirs(train_database_path)
    os.mkdir(f"{train_database_path}/images")
    os.mkdir(f"{train_database_path}/images/train")
    os.mkdir(f"{train_database_path}/images/validation")

    os.mkdir(f"{train_database_path}/labels")
    os.mkdir(f"{train_database_path}/labels/train")
    os.mkdir(f"{train_database_path}/labels/validation")

# create yaml
with open(f"{train_database_path}.yaml", "w") as yaml:
    yaml.write(f"train: {train_database_path}/images/train\n")
    yaml.write(f"val: {train_database_path}/images/validation\n")
    yaml.write(f"nc: 1\n")
    yaml.write(f'names: ["{train_database_name}"]')

# prase json to yolo file
for k, dp in labelme_json_data_pool.items():
    js_list = [i for i in os.listdir(dp) if ".json" in i]
    for js in js_list:
        js_name, _ = os.path.splitext(js)

        json_full_path = os.path.join(dp, js)
        with open(json_full_path, 'r', encoding='utf-8') as f:
            # lines = f.readlines()
            label_data = json.load(f)

        img_relative_path = label_data['imagePath']

        # calculate the abs path of image
        mod_path = Path(json_full_path).parent
        img_abs_path = (mod_path / img_relative_path).resolve()

        img_id = f"{k}_{js_name}"
        _, suffix = os.path.splitext(img_abs_path)
        copyfile(img_abs_path, f'{train_database_path}/images/train/{img_id}{suffix}')
        copyfile(img_abs_path, f'{train_database_path}/images/validation/{img_id}{suffix}')

        img_w = label_data['imageWidth']
        img_h = label_data['imageHeight']

        yolo_data = []

        for object in label_data["shapes"]:
            # only consider bbox labels in this case
            if object['shape_type'] == 'rectangle':
                rectangle = object['points']
                rectangle_ = np.array(rectangle)
                x0, y0 = rectangle_[0, :]
                x1, y1 = rectangle_[1, :]

                xc = (x0 + x1) / 2
                yc = (y0 + y1) / 2

                w = abs(x0 - x1)
                h = abs(y0 - y1)

                xc /= img_w
                w /= img_w
                yc /= img_h
                h /= img_h

                yolo_data.append([0, xc, yc, w, h])

        yolo_data = np.asarray(yolo_data)
        np.savetxt(
            f'{train_database_path}/labels/train/{img_id}.txt',
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        np.savetxt(
            f'{train_database_path}/labels/validation/{img_id}.txt',
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )

print("Training database successfully prepared")