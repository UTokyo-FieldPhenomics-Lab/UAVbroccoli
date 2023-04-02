from config import *

import os
import platform

if not os.path.exists(yolo_model_save_path):
    os.makedirs(yolo_model_save_path)

scripts = f"python ./yolov5/train.py \
--img {yolo_model_image_size} \
--batch {yolo_model_batch_size} \
--epochs {yolo_model_epochs} \
--data {train_database_path}.yaml \
--cfg {yolo_model_structure_config} \
--name {yolo_model_name} \
--project {yolo_model_save_path}"

if platform.system() == "Windows":
    scripts += " --workers 0"

print(">>>" + scripts)
os.system(scripts)