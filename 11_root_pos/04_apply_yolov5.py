from config import *

import os

if not os.path.exists(yolo_apply_results_folder):
    os.makedirs(yolo_apply_results_folder)

dom_name = os.path.split(dom_path_for_detection)[1]
# -> broccoli_tanashi_dom.tif

clip_folder = os.path.join(dom_slice_save_folder, dom_slice_save_format, dom_name)
# -> 10_root_posotion/jpg/broccoli_tanashi_dom.tif/

scripts = f"python ./yolov5/detect.py \
--source {clip_folder} \
--weights {yolo_apply_weights_path} \
--imgsz {yolo_model_image_size} \
--conf-thres {yolo_apply_confidence_threshold} \
--iou-thres {yolo_apply_iou_threshold} \
--save-txt \
--save-conf \
--project {yolo_apply_results_folder}"

print(">>>" + scripts)
os.system(scripts)
