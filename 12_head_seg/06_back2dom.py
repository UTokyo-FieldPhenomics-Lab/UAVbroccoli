from config import *

import random
import shapely
from skimage.transform import PiecewiseAffineTransform
from scipy.spatial import ConvexHull, KDTree
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops, regionprops_table
import cv2

import utils.circle_fit as cf

##############
# data loader
##############

# read the metashape projects and outputs
#-------------------------------------------
ms_dict = {}
for key, val in process_date.items():
    ms = idp.Metashape(val['project'], val['chunk_id'])
    ms.dom = idp.GeoTiff(val['dom'])
    ms.dsm = idp.GeoTiff(val['dsm'])
    ms.crs = ms.dom.crs

    ms_dict[key] = ms

# read the broccoli head position / idx info
#-------------------------------------------
pickle_file = "__pycache__/points_np2d.pkl"    # created by 01_prepare_data
with open(pickle_file, 'rb') as handle:
    ## a nx2 numpy array to record 
    ## all broccoli head root position
    head_np2d = pickle.load(handle)
kdtree = KDTree(head_np2d)

pts_file = "__pycache__/points_dict.pkl"   # created by 06_back2dom_note.ipynb
if os.path.exists(pts_file):
    with open(pts_file, 'rb') as handle:
        head_idx2np_dict = pickle.load(handle)
else:                                      # or created by the first time run to save time
    head_idx2np_dict = idp.shp.read_shp(root_shp, shp_proj=ms_dict[key].crs, name_field=0)
    with open(pts_file, 'wb') as handle:
        pickle.dump(head_idx2np_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

## the index of each line of head_np2d
broccoli_idx = np.asarray(list(head_idx2np_dict.keys()), dtype=np.uint16)

# read roi_grid geo-position
#-------------------------------------------
roi = idp.ROI(f"{project_data_folder}/02_GIS/split_grid.shp", name_field=0)

# read bisenet segment results
#-------------------------------------------
head_segment_json = idp.jsonfile.read_json(f"{result_folder}/out_results.json")

##############
# transformer & calculator
##############

def create_control_points(roi_geo_xy):
    xmin, ymin = roi_geo_xy.min(axis=0)
    xmax, ymax = roi_geo_xy.max(axis=0)

    # pick 70 x 70 grid
    gridx = np.linspace(xmin-0.5, xmax+0.5, 70)
    gridy = np.linspace(ymin-0.5, ymax+0.5, 70)

    xx, yy = np.meshgrid(gridx, gridy)
    xxyy = np.dstack([xx.flat, yy.flat])[0]

    return xxyy

def back2raw_one2one(ms, coords, img_name):
    return ms._back2raw_one2one(
        ms._world2local(ms._crs2world(coords)), 
        img_name
    )

def mk_rm_line(bound_np, rm_bound_id):
    bound_np = bound_np[:-1, :]
    return shapely.geometry.LineString(bound_np[rm_bound_id-1:rm_bound_id+2,:])

def find_polygon_circular_center(polygon):
    if not isinstance(polygon, np.ndarray):
        polygon = np.asarray(polygon)
    hull = ConvexHull(polygon)
    hull_pts = polygon[hull.vertices]
    xy, yc, r, sigma = cf.hyperLSQ(hull_pts)

    return xy, yc, r, sigma

def draw_binary_image(poly_list):
    poly_merge = np.vstack(poly_list)
    
    xmax, ymax = np.max(poly_merge, axis=0)
    xmin, ymin = np.min(poly_merge, axis=0)
    xlen = xmax - xmin
    ylen = ymax - ymin

    res = 0.001 # 1mm/ pix
    
    w = xlen / res
    h = ylen / res

    im = Image.new(mode='1', size=tuple(np.ceil([w, h]).astype(int)))
    draw = ImageDraw.Draw(im)
    
    points = []
    for p in poly_list:
        point = (p - np.asarray([xmin, ymin])) / res
        draw.polygon(point.reshape(len(point)*2).tolist(), fill='white', outline='white')
        points.append(point)
        
    return np.asarray(im), points

def get_2D_traits(head_geo_np, date, this_broccoli_id, xc, yc, r):
    binary_img, pix = draw_binary_image([head_geo_np])
    rect = cv2.minAreaRect(pix[0].astype(np.int32))
    label_img = label(binary_img)

    props = pd.DataFrame(regionprops_table(label_img, 
                        properties=["area", "convex_area", "eccentricity", 
                                    "equivalent_diameter", "major_axis_length", "minor_axis_length", 
                                    "perimeter"]))
    props['circularity'] = 4 * props.area * np.pi / props.perimeter ** 2
    props['date'] = date
    props['label'] = this_broccoli_id

    props["circular_x"] = xc
    props["circular_y"] = yc
    props["circular_r"] = r * 1000  # mm

    props["min_area_rect_max"] = max(rect[1][0], rect[1][1])
    props["min_area_rect_min"] = min(rect[1][0], rect[1][1])

    props['polygon'] = str(head_geo_np.tolist())

    props = props[["date", "label", "area", 
                "convex_area", "eccentricity", "equivalent_diameter", "major_axis_length", 
                "minor_axis_length", "min_area_rect_max", "min_area_rect_min", "perimeter", 
                'circularity', "circular_x", "circular_y", "circular_r", "polygon"]]
    return props

# init result containers
#-------------------------------------------
div_thresh = 0.3 
props_all = pd.DataFrame(columns=["date", "label", "area", "convex_area", "eccentricity", 
                                  "equivalent_diameter", "major_axis_length", "minor_axis_length", 
                                  "min_area_rect_max", "min_area_rect_min", "perimeter", 'circularity',
                                  "circular_x", "circular_y", "circular_r", "polygon"])
stdout = ''
# start the loop, check 06_back2dom_note.ipynb 
#    for more details
#-------------------------------------------
tbar = tqdm(head_segment_json.items())
# head_segment_json['20220405_0']['1_DJI_0356'].keys()
# --> ['out', 'cropLeftTopCorner', 'gridCoordOnCroppedImage', 'headCoordOnCroppedImage']
for date, val in tbar:
    current_date = date[2:-2]   # '20220405_0' -> '220405'
    tbar.set_description_str(f"Processing [{date}] -> {current_date}")

    ms = ms_dict[current_date]
    head_np2d_pix_on_dom = ms.dom.geo2pixel(head_np2d)

    roi_geo_xyz = roi.copy()
    roi_geo_xyz.get_z_from_dsm(ms.dsm, mode='face')

    qbar = tqdm(val.items(), leave=False)
    for img_name, head_json in qbar:
        ins = img_name.split('_')  # '1_DJI_0356' -> [1, DJI, 0356]
        roi_id = ins[0]
        on_img = f"{ins[1]}_{ins[2]}"

        qbar.set_description_str(f"[roi_id]: {roi_id} | [on_img]: {on_img}")

        # get the z values of geo_xy
        ctrl_pts_geo_xyz = idp.ROI()
        ctrl_pts_geo_xyz.crs = roi.crs
        ctrl_pts_geo_xyz[roi_id] = create_control_points(roi[roi_id])

        # roi_geo_xyz = idp.ROI()
        # roi_geo_xyz.crs = roi.crs
        # roi_geo_xyz[roi_id] = roi[roi_id]

        ctrl_pts_geo_xyz.get_z_from_dsm(ms.dsm, mode='point')
        # roi_geo_xyz.get_z_from_dsm(ms.dsm, mode='face')

        # get the pixel coordinates on raw image & dom
        ctrl_pts_pix_on_img = back2raw_one2one(ms, ctrl_pts_geo_xyz[0], on_img)
        ctrl_pts_pix_on_dom = ms.dom.geo2pixel(ctrl_pts_geo_xyz[0])

        roi_pix_on_img = back2raw_one2one(ms, roi_geo_xyz[roi_id], on_img)
        roi_pix_on_dom = ms.dom.geo2pixel(roi_geo_xyz[roi_id])

        # calculate the piecewise affine transformation
        pie_aff_tform = PiecewiseAffineTransform()
        pie_aff_tform.estimate(ctrl_pts_pix_on_dom, ctrl_pts_pix_on_img)

        # apply to head results
        roi_on_cropped_img = np.asarray(
            head_segment_json[date][img_name]['gridCoordOnCroppedImage']
        )
        offset = np.asarray(head_segment_json[date][img_name]['cropLeftTopCorner'])

        grid_poly = shapely.geometry.Polygon(roi_on_cropped_img)
        rm_line = mk_rm_line(roi_on_cropped_img, 2)

        # remove the segment head results outside the ROI
        #  and those touches the bottom & right lines (duplicated with neighour ROI)
        shapely_detect_in = []
        shapely_detect_out = []
        for i, head in enumerate(head_segment_json[date][img_name]['out']):
            head_shply = shapely.geometry.Polygon(head)
            if head_shply.intersects(rm_line):   # rm touches the bottom & right lines
                shapely_detect_out.append(i)
            else:
                if head_shply.intersects(grid_poly):
                    shapely_detect_in.append(i)
                else:
                    shapely_detect_out.append(i)  # rm the outside ROI

        # draw previous figures for checking
        #-------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        buffer = 50
        dom_xmin, dom_ymin = roi_pix_on_dom.min(axis=0)
        dom_xmax, dom_ymax = roi_pix_on_dom.max(axis=0)

        dom_grid_crop = ms.dom.crop_rectangle(
            left = int(dom_xmin - buffer),
            top  = int(dom_ymin - buffer),
            w    = int(dom_xmax - dom_xmin + buffer * 2),
            h    = int(dom_ymax - dom_ymin + buffer * 2),
            is_geo=False   # this is pixel coordinate on dom
        )
        dom_offset = np.array([dom_xmin-buffer, dom_ymin-buffer])
        ax.imshow(dom_grid_crop[:,:,0:3])  # the DOM map
        offset_coord = roi_pix_on_dom - dom_offset
        ax.plot(*offset_coord.T, '--r')   # the ROI boundary

        # show the head position points & id

        rough_idx = (head_np2d_pix_on_dom[:,0] >= (dom_xmin-buffer)) & \
                    (head_np2d_pix_on_dom[:,0] <= (dom_xmax+buffer)) & \
                    (head_np2d_pix_on_dom[:,1] >= (dom_ymin-buffer)) & \
                    (head_np2d_pix_on_dom[:,1] <= (dom_ymax+buffer))
        broccoli_in = broccoli_idx[rough_idx]
        broccoli_in_pos = head_np2d_pix_on_dom[rough_idx] - dom_offset
        ax.scatter(*broccoli_in_pos.T, facecolors='b', edgecolors='w', linewidths=2, s=30, alpha=0.5)

        for bid, binpos in zip(broccoli_in, broccoli_in_pos):
            ax.annotate(bid, xy=binpos)

        save_img = False
        #-------------------------------------------
        # continue in the next forloops

        head_on_dom_list = []
        for in_id in shapely_detect_in:
            head_np = np.asarray(head_segment_json[date][img_name]['out'][in_id])
            head_pix_np = pie_aff_tform.inverse(head_np + offset)

            head_geo_np = ms.dom.pixel2geo(head_pix_np)
            xc, yc, r, sigma = find_polygon_circular_center(head_geo_np)

            dist, id_list_idx = kdtree.query(np.array([xc, yc]))
            this_broccoli_id = broccoli_idx[id_list_idx]

            if dist > div_thresh:
                color = "r"
                stdout += f"Broccoli {this_broccoli_id} detect distence {dist} is over {div_thresh}\n"
                save_img = True
            else:
                color = "g"

            # draw previous figures for checking
            #-------------------------------------------
            bb = head_pix_np - dom_offset
            ax.plot(*bb.T, '-r', alpha=0.5)

            xc_plot, yc_plot, r_plot, _ = find_polygon_circular_center(
                head_pix_np - dom_offset
            )

            ax.add_patch(
                plt.Circle((xc_plot, yc_plot), r_plot, color='b', fill=False)
            )

            # draw the line between broccoli id -> circular center
            head_pix_xy = head_np2d_pix_on_dom[id_list_idx] - dom_offset
            ax.plot([xc_plot, head_pix_xy[0]], [yc_plot, head_pix_xy[1]], c=color, alpha=0.5)
            #-------------------------------------------\

            props = get_2D_traits(head_geo_np, date, this_broccoli_id, xc, yc, r)
            props_all.loc[len(props_all)] = props.iloc[0]
            
        plt.axis('off')
        plt.tight_layout()
        if random.randint(0, 10) == 5 or save_img:
            plt.savefig(f"{project_data_folder}/{working_spacename}/results/back2dom_match/{save_img}_{date}_{img_name}.png")
        plt.clf()
        plt.cla()
        plt.close()

    props_all.to_excel(f"{project_data_folder}/{working_spacename}/results/props_all_2022.xlsx", index=False)

print(stdout)

# todo:
# add distance into traits?
# fix circlar errors