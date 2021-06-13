from config import *

# plots = ['0518_p', '0520_p', '0522_m', '0528_p']
#plots = ['0520_p', '0522_m', '0528_p']
plots = ['0525_p', '0526_p']

for p in plots:
    cp = Paths(p)
    
    print(f"\n++++++++++{cp.project_name}+++++++++++++")
    
    root_pos = pd.read_csv(f"{cp.root}/10_locate_by_cv/color_label_0417_mavic/keep_bbox_cp.csv", index_col=0)
    root_select = root_pos[root_pos.fid >= 0]
    
    print("\n[08 batch] root position loaded")
    
    keep_bbox, rm_bbox = read_label(cp.project_name, 'ins_bg')
    
    print("\n[08 batch] DL bbox loaded")
    
    fig_path = f"{cp.root}/11_instance_seg/detect+bg.problem/{cp.project_name}"
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    
    draw_bbox_points_miss_overview(root_select, keep_bbox, fig_title=cp.project_name, fig_path=fig_path)
    
    print("\n[08 batch] overall plot saved")
    
    point_wrong, bbox_wrong = find_bbox_points_not_match(root_select, keep_bbox, points_buffer=0.1)
    point_wrong.to_csv(f"{fig_path}/true_negative({len(point_wrong)}).csv")
    bbox_wrong.to_csv(f"{fig_path}/false_positive({len(bbox_wrong)}).csv")
    
    print("\n[08 batch] wrong result csv file saved")
    
    p4d = Pix4D(project_path=cp.pix4d_project, 
                raw_img_path=cp.raw_img, 
                project_name=cp.project_name,
                param_folder=cp.pix4d_param)
    
    grid_len = 1300
    buffer_len = 200
    ts = TiffSpliter(tif_path=p4d.dom_file, grid_h=grid_len, grid_w=grid_len, grid_buffer=buffer_len)
    
    print("\n[08 batch] start save individual images")
    draw_bbox_points_miss_individual(p4d, ts, root_select, keep_bbox, point_wrong, bbox_wrong, fig_path, neighbour_buffer = 0.5)

    