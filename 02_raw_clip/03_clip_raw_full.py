from config import *
from easyric.io.json import dict2json

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

        #root = shapefile.Reader(f"{p2.root}/10_locate_by_cv/color_label_0417_mavic/keep_points_manual.shp")
        #root = shapefile.Reader(f"{p2.root}/12_locate_by_yolo/sorted_id.shp")
        
        #points_np = np.zeros((0,2))
        #for i, point in enumerate(root.shapes()):
        #    points_np = np.vstack([points_np, np.asarray(point.points)])
            
        #ht = geotiff.mean_values(p4d.dsm_file)
        #points_np3d = np.insert(points_np, 2, ht, axis=1)
        
        
        points = shp.read_shp3d(f"{p2.root}/12_locate_by_yolo/sorted_id.shp", p4d.dsm_file, get_z_by="max", get_z_buffer=0.2, geo_head=p4d.dsm_header)
        points_np3d = np.zeros((0,3))
        for k, p in points.items():
            points_np3d = np.vstack([points_np3d, p])
        
        deeplab_dict = {}
        for img in p4d.img:
            points_raw = geo2raw.pmatrix_calc(p4d, points_np3d-p4d.offset.np, img.name, distort_correct=True)

            points_left = points_raw[(points_raw[:,0] > 0) & (points_raw[:,0] < img.w) & 
                                     (points_raw[:,1] > 0) & (points_raw[:,1] < img.h), :]

            if len(points_left) > 0:
                deeplab_dict[img.name] = {"imagePath": img.path,
                                          "points": points_left.tolist()}
                
        dict2json(deeplab_dict, f"{p2.root}/11_labelme_json/{p2.project_name}.json")