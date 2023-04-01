###########
# 00 init #
###########
project_data_folder = "Z:/hwang_Pro/data/2022_tanashi_broccoli"
working_spacename = "12_head_segment"

###################
# 01 prepare data #
###################
root_shp = f"{project_data_folder}/11_root_position/shapefiles/ordered_center_points.shp"
grid_shp = f"{project_data_folder}/02_GIS/split_grid.shp"
project_outputs_folder = f"{project_data_folder}/01_metashape_projects/outputs"
crop_image_size = 1000

process_date = {
    # date_id: {
    #    project: ''
    #    dom: ''
    #    dsm: ''
    # }
    "220321": {
        "project": f"{project_data_folder}/01_metashape_projects/broccoli_autumn21.psx", 
        "chunk_id": "20220321_0",
        "dom": f"{project_outputs_folder}/broccoli_tanashi_5_20220321_0/broccoli_tanashi_5_20220321_0_dom.tif",
        "dsm": f"{project_outputs_folder}/broccoli_tanashi_5_20220321_0/broccoli_tanashi_5_20220321_0_dsm.tif"
        },
    "220325": {
        "project": f"{project_data_folder}/01_metashape_projects/broccoli_autumn21.psx",
        "chunk_id": "20220325_0",
        "dom": f"{project_outputs_folder}/broccoli_tanashi_5_20220325_0/broccoli_tanashi_5_20220325_0_dom.tif",
        "dsm": f"{project_outputs_folder}/broccoli_tanashi_5_20220325_0/broccoli_tanashi_5_20220325_0_dsm.tif"
        },
    "220329": {
        "project": f"{project_data_folder}/01_metashape_projects/broccoli_autumn21.psx",
        "chunk_id": "20220329_0",
        "dom": f"{project_outputs_folder}/broccoli_tanashi_5_20220329_0/broccoli_tanashi_5_20220329_0_dom.tif",
        "dsm": f"{project_outputs_folder}/broccoli_tanashi_5_20220329_0/broccoli_tanashi_5_20220329_0_dsm.tif"
        },
    "220331": {
        "project": f"{project_data_folder}/01_metashape_projects/broccoli_autumn21.psx",
        "chunk_id": "20220331_0",
        "dom": f"{project_outputs_folder}/broccoli_tanashi_5_20220331_0/broccoli_tanashi_5_20220331_0_dom.tif",
        "dsm": f"{project_outputs_folder}/broccoli_tanashi_5_20220331_0/broccoli_tanashi_5_20220331_0_dsm.tif"
        },
    "220405": {
        "project": f"{project_data_folder}/01_metashape_projects/broccoli_autumn21.psx",
        "chunk_id": "20220405_0",
        "dom": f"{project_outputs_folder}/broccoli_tanashi_5_20220405_0/broccoli_tanashi_5_20220405_0_dom.tif",
        "dsm": f"{project_outputs_folder}/broccoli_tanashi_5_20220405_0/broccoli_tanashi_5_20220405_0_dsm.tif"
    },
}



import numpy as np
import easyidp as idp
import os
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm