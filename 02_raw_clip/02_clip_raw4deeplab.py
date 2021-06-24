from config import *
from PIL import Image
from easyric.io.json import dict2json


###############  
# 01_clip_raw #
###############

# please refer to 01_slice/09_project2raw.ipynb
# calculate the distance
def calculate_dist2center(p4d, geo2raw_out_dict, id_name):
    dist_container = pd.DataFrame(columns=['id', 'image', 'xc', 'yc', 'dist', 'angle', #'direction', 
                                           "select", 'roi'])
    for i, c in geo2raw_out_dict.items():
        c = np.asarray(c)
        #x0, y0 = c.mean(axis=0)
        xmin, ymin = c.min(axis=0)
        xmax, ymax = c.max(axis=0)
        roi_w = xmax - xmin
        roi_h = ymax - ymin
        x0 = (xmax+xmin)/2
        y0 = (ymax+ymin)/2
                
        if roi_w > 1500 or roi_h > 1500:
            print(f"[Warning]: plot [{id_name}] on img [{i}], roi size ({roi_w},{roi_h}) exceed (1500, 1500)")
        

        x1, y1 = 0.5 * p4d.img[i].w, 0.5 * p4d.img[i].h
        d = np.sqrt((x1-x0) ** 2 + (y1 - y0) ** 2)
        
        v1 = np.asarray([x0-x1, y0-y1])
        v0 = np.asarray([0, 1])
        
        # https://blog.csdn.net/qq_32424059/article/details/100874358
        angle = calc_angle_2(v0, v1)
        """
        if angle > 45 and angle <= 135:
            direction = 'Right'
        elif angle > 135 and angle <= 225:
            direction = "Down"
        elif angle > 225 and angle <= 315:
            direction = "Left"
        else:
            direction = "Up"
        """
        
        dist_container.loc[len(dist_container)] = {"id":id_name, "image":i, "xc":x0, "yc":y0, "dist":d, 
                                                   "angle":angle, #"direction":direction,
                                                   "select":False, "roi":c}
        dist_container['dist'] = dist_container['dist'].astype(float)

    return dist_container

def calc_angle_2(v1, v2):
    '''
    支持大于180度计算
    https://www.pythonf.cn/read/131921
    '''
    r = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)))
    deg = r * 180 / np.pi

    a1 = np.array([*v1, 0])
    a2 = np.array([*v2, 0])

    a3 = np.cross(a1, a2)

    if np.sign(a3[2]) > 0:
        deg = 360 - deg

    return deg

def select_best(dist_container, strategy="min_dist"):
    if strategy == "min_dist":
        dist_container = dist_container.sort_values(by=['dist'], axis=0, ascending=True)
        dist_container = dist_container.reset_index()
    
        return dist_container.iloc[0:10, :]
    else:
        dist_container = dist_container.sort_values(by=['direction', 'dist'], axis=0, ascending=True)
        
        direct = ['Up', 'Right', 'Down', 'Left']

        for i, d in enumerate(direct):
            t_all = reverse_dist[reverse_dist.direction == d]
            t = t_all.iloc[0,:]
            ax.scatter(t.xc, t.yc, c=color[i])
            
            

if __name__ == "__main__":
    #todo_pool = ["0520_p", "0522_p", "0525_p", "0526_p", "0528_p"]
    todo_pool = ["210512", "210514", "210515", "210519", "210520", "210526"]
    
    for tp in todo_pool:
        #p2 = Paths(tp)
        p2 = Paths(tp, year=2021)

        p4d = Pix4D(project_path=p2.pix4d_project, 
                    raw_img_path=p2.raw_img, 
                    project_name=p2.project_name,
                    param_folder=p2.pix4d_param)

        #shp_file = r"Y:\hwang_Pro\data\2020_tanashi_broccoli\02_GIS\rotate_grids\split_grid_2.5m.shp"
        shp_file = f"{p2.root}/02_GIS/split_grid.shp"

        process_area = shp.read_shp3d(shp_file, dsm_path=p4d.dsm_file, geotiff_proj=p4d.dsm_header['proj'], name_field="id", get_z_by="mean")

        # calculate clipped raw image sectors
        result_container = pd.DataFrame(columns=['id', 'image', 'xc', 'yc', 'dist', 'angle', 
                                              "select", 'roi'])

        for plot_id, roi in process_area.items():
            img_dict = geo2raw.get_img_coords_dict(p4d, roi-p4d.offset.np, method="pmat")

            reverse_dist = calculate_dist2center(p4d, img_dict, id_name=plot_id)

            # filter 3 closest raw images
            selected_idx = reverse_dist.copy().sort_values(by=['dist'], axis=0, ascending=True).index[0:3]
            reverse_dist.loc[selected_idx, 'select'] = True

            result_container = pd.concat([result_container, reverse_dist])

        result_container['offset_x'] = round(result_container.xc - 750).astype(np.int32)
        result_container['offset_y'] = round(result_container.yc - 750).astype(np.int32)


        csv_folder = f"{p2.root}/13_roi_on_raw/{p2.project_name}"
        if not os.path.exists(csv_folder):
            os.mkdir(csv_folder)

        result_container.to_csv(f"{csv_folder}.csv", index=False)


        # read broccoli root shp file
        
        #root = shapefile.Reader(f"{p2.root}/10_locate_by_cv/color_label_0417_mavic/keep_points_manual.shp")
        root = shapefile.Reader(f"{p2.root}/12_locate_by_yolo/sorted_id.shp")
        
        points_np = np.zeros((0,2))
        for i, point in enumerate(root.shapes()):
            points_np = np.vstack([points_np, np.asarray(point.points)])

        deeplab_dict = {}
        for idx, row in result_container[result_container.select].iterrows():
            original = Image.open(p4d.img[row.image].path)
            cropped = original.crop([row.offset_x, row.offset_y, row.offset_x+1500, row.offset_y+1500])

            points_np3d = np.insert(points_np, 2, process_area[row.id][0,2], axis=1)
            points_raw = geo2raw.pmatrix_calc(p4d, points_np3d-p4d.offset.np, row.image, distort_correct=True)

            points_left = points_raw[(points_raw[:,0] > row.offset_x) & (points_raw[:,0] < row.offset_x+1500) & 
                                     (points_raw[:,1] > row.offset_y) & (points_raw[:,1] < row.offset_y+1500), :]

            print(row.image, row.id, len(points_left), end='\r')
            points_left_offset = points_left - np.asarray([[row.offset_x, row.offset_y]])

            img_name= f"{row.id}_{row.image}"

            deeplab_dict[img_name] = {"imagePath": f"./{p2.project_name}/{img_name}",
                                      "points": points_left_offset.tolist()}

            cropped.save(f"{csv_folder}/{row.id}_{row.image}")

        dict2json(deeplab_dict, f"{p2.root}/13_roi_on_raw/{p2.project_name}.json")