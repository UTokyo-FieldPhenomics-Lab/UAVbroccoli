###########
# 00 init #
###########
easyidp_package_path = "Z:/hwang_Pro/github/EasyIDP"
project_data_folder = "E:/2022_tanashi_broccoli"
working_spacename = "11_root_position"

################
# 01 slice dom #
################
dom_path_for_detection = f"{project_data_folder}/01_metashape_projects/outputs/broccoli_tanashi_5_20211101_0_dom.tif"
dom_slice_save_folder = f"{project_data_folder}/{working_spacename}"
dom_slice_save_format = "jpg"
dom_slice_length = 1300   # must be a square
dom_slice_buffer = 200


############################
# 02 prepare yolo database #
############################
labelme_json_data_pool = {
    # data_id: folder contains labelme json
    "2021a":f"{project_data_folder}/{working_spacename}/labelme_json",  # 2021 autumn train data
    "2021s":"Z:/hwang_Pro/data/2021_tanashi_broccoli/11_instance_seg/yolo_json", # 2021 summer train data
    "2020":"Z:/hwang_Pro/data/2020_tanashi_broccoli/11_instance_seg/yolo_json", # 2020 train data
    "2019":"Z:/hwang_Pro/data/2019_tanashi_broccoli5/11_instance_seg/yolo_json",  #2019 train data
}
train_database_path = f"{project_data_folder}/{working_spacename}/yolo_train/training_database"
# optional
train_database_name = "broccoli_root"


#######################
# 03 train yolo model #
#######################
yolo_model_image_size = 1500  # optional
yolo_model_batch_size = 8
yolo_model_epochs = 300
yolo_model_structure_config = "../yolov5/models/yolov5s.yaml"
yolo_model_name = "br"
yolo_model_save_path = f"{project_data_folder}/{working_spacename}/yolo_train/runs"


#######################
# 04 apply yolo model #
#######################
yolo_apply_results_folder = f"{project_data_folder}/{working_spacename}/yolo_results"

# please check carefully of model ouptuts, need change every time when run scripts!
yolo_apply_weights_path = f"{yolo_model_save_path}/br/weights/best.pt"

yolo_apply_confidence_threshold = 0.3  # please edit this according to your results
yolo_apply_iou_threshold = 0


########################
# 05 yolo to shapefile #
########################
# please use QGIS to make a rectange bounding box of fields (very tight to crops)
field_rectange_bounding_box_shapefile = f"{project_data_folder}/02_GIS/plot_edge_for_yolo.shp"
# please check carefully about this:
yolo_apply_label_folder = f"{yolo_apply_results_folder}/exp/labels"
shapefile_save_folder = f"{project_data_folder}/{working_spacename}/shapefiles"


#####################
# 06 order by ridge #
#####################
edited_center_points_shapefile = f"{shapefile_save_folder}/edited_center_points.shp"
ridge_strength_ratio = 10
ridge_direction = "x"  # if wrong direction, change to "y"
ridge_distance_parameters = 3 # please decrease if some ridge is missing
ridge_height_parameters = 20  # please increase if some bottom small ridge detect as ridge

# regression points to ridge lines
ridge_buffer = 0.5   # default is the middle line of two ridges
#      rg1          rg2
# |    [|]    |     [|]
# |<---[|]--->|     [|]
# |    [|] 0.5|     [|]
# |   buffer  |     [|]
use_ransac = False
ransac_residual_threshold=35  # more crops in a ridge, more values
ransac_max_trials=2000

# save files
figure_save_path = shapefile_save_folder

import sys
sys.path.insert(0, easyidp_package_path)
from easyric.caas_lite import TiffSpliter
from easyric.objects import Pix4D
from easyric.io import shp, geotiff, plot
from easyric.calculate import geo2raw, geo2tiff
import shapefile

def save_shp(df, shp_path, type='bbox'):
    with shapefile.Writer(shp_path) as shp:
        shp.field('name', 'C')
        for idx, row in df.iterrows():
            if type=='bbox':
                # only suits for pandas with ['x0_geo', 'x1_geo', 'y0_geo', 'y1_geo'] keys
                x0 = row.x0_geo
                y0 = row.y0_geo
                x1 = row.x1_geo
                y1 = row.y1_geo

                coord = [[x0, y0],[x1, y0],[x1, y1],[x0, y1],[x0, y0]]
                shp.poly([coord])
                shp.record(str(idx))
            elif type=='points':
                # only suits for pandas with ['x_geo', 'y_geo', 'fid'] keys
                shp.point(row.x_geo, row.y_geo)
                shp.record(str(int(row.fid)))