import sys
sys.path.insert(0, f'D:/OneDrive/Program/GitHub/EasyIDP')

import os
import shapefile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easyric.caas_lite import TiffSpliter
from easyric.objects import Pix4D
from easyric.io import geotiff

from dom_label2raw import read_label

def save_shp(bbox, shp_path):
    with shapefile.Writer(shp_path) as shp:
        shp.field('name', 'C')
        for i in range(len(bbox)):
            x0 = bbox.x0_geo.iloc[i]
            y0 = bbox.y0_geo.iloc[i]
            x1 = bbox.x1_geo.iloc[i]
            y1 = bbox.y1_geo.iloc[i]

            coord = [[x0, y0],[x1, y0],[x1, y1],[x0, y1],[x0, y0]]
            shp.poly([coord])
            shp.record(str(i))

if __name__ == '__main__':
    project_names = ['broccoli_tanashi_5_20200528_P4M_10m_before',
                     'broccoli_tanashi_5_20200526_P4M_10m_before',
                     'broccoli_tanashi_5_20200522_P4M_10m_before',
                     'broccoli_tanashi_5_20200522_mavicRGB_15m_M_before']

    shp_folder = "Z:/共享云端硬盘/broccoliProject/pred_bbox_nms_0.3/bbox_shp"
    
    for project_name in project_names:
        keep_bbox, rm_bbox = read_label(project_name)
        
        kp_shp_path = f"{shp_folder}/{project_name}_keep"
        rm_shp_path = f"{shp_folder}/{project_name}_remove"
        
        save_shp(keep_bbox, kp_shp_path)
        save_shp(rm_bbox, rm_shp_path)