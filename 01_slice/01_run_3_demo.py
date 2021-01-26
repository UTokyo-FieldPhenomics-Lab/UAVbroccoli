import numpy as np

import sys
import os
sys.path.insert(0, f'D:/OneDrive/Program/GitHub/EasyIDP')

from easyric.objects import Pix4D
from easyric.caas_lite import TiffSpliter

def one_plot(project_name, drive='Z'):
    project_path = f"{drive}:/共享云端硬盘/broccoliProject/tanashi_broccoli2020_RGB_AP/{project_name}"
    raw_img_folder = f"{drive}:/共享云端硬盘/broccoliProject/rgb_raw/{project_name}"
    param_folder = f"{project_path}/params"

    out_folder = f"{drive}:/共享云端硬盘/broccoliProject/anotation_use/jpg/{project_name}"
    json_name = f"{out_folder}.json"
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    p4d = Pix4D(project_path=project_path, 
                raw_img_path=raw_img_folder, 
                project_name=project_name,
                param_folder=param_folder)
    
    ts = TiffSpliter(tif_path=p4d.dom_file, grid_h=1300, grid_w=1300, grid_buffer=200)
    
    offset_json = {}
    for w_id, w_st in enumerate(ts.wgrid_st):
        for h_id, h_st in enumerate(ts.hgrid_st):
            tiff_name = ts.id2name(w_id=w_id, h_id=h_id)
            offset_json[tiff_name] = {'x':ts.wgrid_st[w_id], 
                                      'y':ts.hgrid_st[h_id]}
    ts.dict2json(offset_json, json_name)
    
    ts.save_all_grids(save_folder=out_folder, extend=True, skip_empty=True, format='jpg')
    

if __name__ == '__main__':
    project_names = ['broccoli_tanashi_5_20200528_P4M_10m_before',
                     'broccoli_tanashi_5_20200526_P4M_10m_before',
                     'broccoli_tanashi_5_20200522_P4M_10m_before',
                     'broccoli_tanashi_5_20200522_mavicRGB_15m_M_before']
    
    for project_name in project_names:
        one_plot(project_name)
