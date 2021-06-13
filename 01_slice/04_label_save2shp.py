from config import *

project_names = ['broccoli_tanashi_5_20200528_P4M_10m_before',
                 #'broccoli_tanashi_5_20200526_P4M_10m_before',
                 #'broccoli_tanashi_5_20200522_P4M_10m_before',
                 'broccoli_tanashi_5_20200522_mavicRGB_15m_M_before']

# shp_folder = "Z:/共享云端硬盘/broccoliProject/pred_bbox_nms_0.3/bbox_shp"

for project_name in project_names:
    cp = Paths(project_name)
    
    # run for dected
    keep_bbox, rm_bbox = read_label(project_name, label="ins")

    kp_shp_path = f"{cp.root}/11_instance_seg/detect.shp/{project_name}_keep"
    rm_shp_path = f"{cp.root}/11_instance_seg/detect.shp/{project_name}_remove"

    save_shp(keep_bbox, kp_shp_path)
    save_shp(rm_bbox, rm_shp_path)
    
    # run for detect+bg
    keep_bbox, rm_bbox = read_label(project_name, label="ins_bg")

    kp_shp_path = f"{cp.root}/11_instance_seg/detect+bg.shp/{project_name}_keep"
    rm_shp_path = f"{cp.root}/11_instance_seg/detect+bg.shp/{project_name}_remove"

    save_shp(keep_bbox, kp_shp_path)
    save_shp(rm_bbox, rm_shp_path)