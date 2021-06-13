import sys
import platform

print("""The following modules are loaded:
import sys
import platform
""")

#################
# get disk code #
#################
user = platform.node()

if user == 'NERV':
    p_disk = "D:" # package disk
    g_disk = "Z:" # google drive disk
elif user == "NERV_SURFACE":
    p_disk = "C:"
    g_disk = "Z:"
else:
    print(f"please add new user [{user}] setting in configs.py")

sys.path.insert(0, f'{p_disk}/OneDrive/Program/GitHub/EasyIDP')


###################
# import packages #
###################
from easyric.caas_lite import TiffSpliter
from easyric.objects import Pix4D
from easyric.io import shp, geotiff, plot
from easyric.calculate import geo2raw, geo2tiff

print("""from easyric.caas_lite import TiffSpliter
from easyric.objects import Pix4D
from easyric.io import shp, geotiff, plot
from easyric.calculate import geo2raw, geo2tiff
""")


class Paths():
    
    def __init__(self, key):
        
        self.root = f"{g_disk}/共享云端硬盘/broccoliProject"
        
        self.project_name = self.generate_name(key)
        
        self.raw_img =f"{self.root}/00_rgb_raw/{self.project_name}"
        self.pix4d_project = f"{self.root}/01_tanashi_broccoli2020_RGB_AP/{self.project_name}"
        self.pix4d_param = f"{self.pix4d_project}/params"
        
        self.ins_label = f"{self.root}/11_instance_seg/detect/{self.project_name}/labels"
        self.ins_label_bg = f"{self.root}/11_instance_seg/detect+bg/{self.project_name}/labels"
        self.sem_label = f"{self.root}/12_sematic_seg/seg_result/{self.project_name}/{self.project_name}.pickle"

        
    @staticmethod
    def generate_name(key):
        
        all_date = {"0313_m"  :'broccoli_tanashi_5_20200313_mavicRGB_15m_M', 
                    "0318_m"  :'broccoli_tanashi_5_20200318_mavicRGB_15m_M', 
                    "0326_m"  :'broccoli_tanashi_5_20200326_mavicRGB_15m_M', 
                    "0327_m"  :'broccoli_tanashi_5_20200327_mavicRGB_15m_M', 
                    "0327_x"  :'broccoli_tanashi_5_20200327_x4sRGB_15m_M', 
                    "0331_m"  :'broccoli_tanashi_5_20200331_mavicRGB_15m_M', 
                    "0406_m"  :'broccoli_tanashi_5_20200406_mavicRGB_15m_M', 
                    "0415_m"  :'broccoli_tanashi_5_20200415_mavicRGB_15m_M', 
                    "0417_p"  :'broccoli_tanashi_5_20200417_P4RGB_15m_M', 
                    "0417_m"  :'broccoli_tanashi_5_20200417_mavicRGB_15m_M', 
                    "0421_m"  :'broccoli_tanashi_5_20200421_mavicRGB_15m_M', 
                    "0422_m"  :'broccoli_tanashi_5_20200422_mavicRGB_15m_M', 
                    "0427_m"  :'broccoli_tanashi_5_20200427_mavicRGB_15m_M', 
                    "0430_m"  :'broccoli_tanashi_5_20200430_mavicRGB_15m_M', 
                    "0508_m"  :'broccoli_tanashi_5_20200508_mavicRGB_15m_M', 
                    "0512_p"  :'broccoli_tanashi_5_20200512_P4M_10m', 
                    "0514_p"  :'broccoli_tanashi_5_20200514_P4M_10m', 
                    "0514_m"  :'broccoli_tanashi_5_20200514_mavicRGB_15m_M', 
                    "0518_p"  :'broccoli_tanashi_5_20200518_P4M_10m', 
                    "0520_p"  :'broccoli_tanashi_5_20200520_P4M_10m', 
                    "0522_p"  :'broccoli_tanashi_5_20200522_P4M_10m_after', 
                    #"0522_p_b":'broccoli_tanashi_5_20200522_P4M_10m_before',  # dom not good
                    "0522_m"  :'broccoli_tanashi_5_20200522_mavicRGB_15m_M_before', 
                    "0525_p"  :'broccoli_tanashi_5_20200525_P4M_10m', 
                    "0525_m"  :'broccoli_tanashi_5_20200525_mavicRGB_15m_M', 
                    "0526_p"  :'broccoli_tanashi_5_20200526_P4M_10m_after', 
                    #"0526_p_b":'broccoli_tanashi_5_20200526_P4M_10m_before',   # dom not good
                    "0528_p"  :'broccoli_tanashi_5_20200528_P4M_10m_before'}
        
        if key in all_date.values():
            return key
        else:
            return all_date[key]

####################        
# Useful functions #
####################
import os
import shapefile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import tifffile

print("""import os
import shapefile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import tifffile
""")

#==============
# 01_exp_slice
#==============

def slice_one_plot(project_name, drive='Z', format='jpg'):
    #project_path = f"{drive}:/共享云端硬盘/broccoliProject/01_tanashi_broccoli2020_RGB_AP/{project_name}"
    #raw_img_folder = f"{drive}:/共享云端硬盘/broccoliProject/00_rgb_raw/{project_name}"
    #param_folder = f"{project_path}/params"
    #out_folder = f"{drive}:/共享云端硬盘/broccoliProject/10_anotation_use/{format}/{project_name}"
    #json_name = f"{out_folder}.json"
    cp = Paths(project_name)
    
    project_path = cp.pix4d_project
    raw_img_folder = cp.raw_img
    param_folder = cp.pix4d_param
    if format=='tif':
        out_folder = f"{cp.root}/10_anotation_use/geotiff/{cp.project_name}"
    else:
        out_folder = f"{cp.root}/10_anotation_use/{format}/{cp.project_name}"
    json_name = f"{out_folder}.json"
    
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    p4d = Pix4D(project_path=project_path, 
                raw_img_path=raw_img_folder, 
                project_name=cp.project_name,
                param_folder=param_folder)
    
    ts = TiffSpliter(tif_path=p4d.dom_file, grid_h=1300, grid_w=1300, grid_buffer=200)
    
    offset_json = {}
    for w_id, w_st in enumerate(ts.wgrid_st):
        for h_id, h_st in enumerate(ts.hgrid_st):
            tiff_name = ts.id2name(w_id=w_id, h_id=h_id)
            offset_json[tiff_name] = {'x':ts.wgrid_st[w_id], 
                                      'y':ts.hgrid_st[h_id]}
    ts.dict2json(offset_json, json_name)
    
    ts.save_all_grids(save_folder=out_folder, extend=True, skip_empty=True, format=format)


#==================
# 04_dom_label2shp
#==================

def read_label(project_name, label='ins', log=True):
    # project_path = f"Z:/共享云端硬盘/broccoliProject/tanashi_broccoli2020_RGB_AP/{project_name}"
    # raw_img_folder=f"Z:/共享云端硬盘/broccoliProject/rgb_raw/{project_name}"
    # label_folder = f"Z:/共享云端硬盘/broccoliProject/pred_bbox_nms_0.3/{project_name}/labels"
    # param_folder = f"{project_path}/params"
    cp = Paths(project_name)
    
    project_path = cp.pix4d_project
    raw_img_folder = cp.raw_img
    
    if label=='ins':
        label_folder = cp.ins_label
    elif label=='ins_bg':
        label_folder = cp.ins_label_bg
    else:
        raise ValueError('only [ins] and [ins_bg] are supported')
    param_folder = cp.pix4d_param

    
    grid_len = 1300
    buffer_len = 200
    
    p4d = Pix4D(project_path=project_path, 
            raw_img_path=raw_img_folder, 
            project_name=project_name,
            param_folder=param_folder)
    
    ts = TiffSpliter(tif_path=p4d.dom_file, grid_h=grid_len, grid_w=grid_len, grid_buffer=buffer_len)
    
    
    # start reading
    bbox_pd = pd.DataFrame(columns=['offset_x', 'offset_y', 
                                    'xc', 'yc', 'w', 'h', 'thresh'])

    print(f"======{project_name}======")
    for label_txt in os.listdir(label_folder):
        if log: print(f"reading {label_txt}", end="\r")
        with open(f"{label_folder}/{label_txt}") as f:
            x_id, y_id = ts.name2id(label_txt, 'txt')
            offset_x, offset_y = ts.wgrid_st[x_id], ts.hgrid_st[y_id]
            for l in f.readlines():
                _, xc, yc, w, h, thresh = l.split(' ')

                bbox_pd.loc[len(bbox_pd),:] = [offset_x, offset_y, 
                                               float(xc), float(yc),
                                               float(w), float(h), thresh]
                
    bbox = bbox_pd.astype(np.float32)
    bbox['xc'] = bbox['xc'] * (grid_len + buffer_len)
    bbox['yc'] = bbox['yc'] * (grid_len + buffer_len)
    bbox['w']  = bbox['w'] * (grid_len + buffer_len)
    bbox['h']  = bbox['h'] * (grid_len + buffer_len)

    bbox['x0'] = bbox['xc'] - bbox['w'] / 2
    bbox['x1'] = bbox['xc'] + bbox['w'] / 2
    bbox['y0'] = bbox['yc'] - bbox['h'] / 2
    bbox['y1'] = bbox['yc'] + bbox['h'] / 2
    
    bbox = bbox.round(0)
    bbox = bbox.astype(np.uint16)
    bbox['thresh'] = bbox_pd['thresh'].astype(np.float32)
    
    bbox['xc_dom'] = bbox['xc'] + bbox['offset_x']
    bbox['yc_dom'] = bbox['yc'] + bbox['offset_y']
    bbox['x0_dom'] = bbox['x0'] + bbox['offset_x']
    bbox['x1_dom'] = bbox['x1'] + bbox['offset_x']
    bbox['y0_dom'] = bbox['y0'] + bbox['offset_y']
    bbox['y1_dom'] = bbox['y1'] + bbox['offset_y']
    
    bbox[['xc_geo', 'yc_geo']]  = geotiff.pixel2geo(bbox[['xc_dom', 'yc_dom']].to_numpy(), p4d.dom_header)
    bbox[['x0_geo', 'y0_geo']]  = geotiff.pixel2geo(bbox[['x0_dom', 'y0_dom']].to_numpy(), p4d.dom_header)
    bbox[['x1_geo', 'y1_geo']]  = geotiff.pixel2geo(bbox[['x1_dom', 'y1_dom']].to_numpy(), p4d.dom_header)
    
    keep_id = nms(bbox[['x0_dom', 'y0_dom', 'x1_dom', 'y1_dom', 'thresh']].to_numpy(), 0.0)
    bbox['keep'] = False
    bbox.loc[keep_id, 'keep'] = True
    
    keep_bbox = bbox[bbox.keep]
    rm_bbox = bbox[-bbox.keep]
    
    return keep_bbox, rm_bbox
    
    
def nms(arr, thresh):
    # reference: https://zhuanlan.zhihu.com/p/128125301
    
    # 首先数据赋值和计算对应矩形框的面积
    # arr的数据格式是arr = [[ xmin, ymin, xmax, ymax,scores]....]

    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    score = arr[:, 4]

    # 所有矩形框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    order = score.argsort()[::-1]
    # 上面这两句比如分数score = [0.72 0.8  0.92 0.72 0.81 0.9 ]
    # 对应的索引order = [2, 5, 4, 1, 3, 0]记住是取出索引，scores列表没变。

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []

    # order会剔除遍历过的方框，和合并过的方框
    while order.size > 0:
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的，就是取最大score的索引
        i = order[0]

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 这边要注意，如果两个方框相交，xx2-xx1和yy2-yy1是正的。
        # 如果两个方框不相交，xx2-xx1和yy2-yy1是负的，我们把不相交的w和h设为0.
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        inter = w * h

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = inter / (areas[i] + areas[order[1:]] - inter)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        index = np.where(ious <= thresh)[0]

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框（n个框计算n-1个IOU值），所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        order = order[index+1]

    return keep

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
                shp.record(str(row.fid))
                
#=================
# 08_find_miss_dl
#=================
def draw_bbox_points_miss_overview(points_pd, bbox_pd, fig_title, fig_path):
    fig, ax = plt.subplots(1,1, figsize=(20,20), dpi=600)

    ax.scatter(points_pd.x_geo, points_pd.y_geo, s=1, c='r')

    for idx, row in bbox_pd.iterrows():
        ax.plot([row.x0_geo, row.x1_geo, row.x1_geo, row.x0_geo, row.x0_geo],
                [row.y0_geo, row.y0_geo, row.y1_geo, row.y1_geo, row.y0_geo],
                'b-', linewidth=0.5)

    ax.set_title(fig_title)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(fig_path)

def find_bbox_points_not_match(points_pd, bbox_pd, points_buffer=0.04, neighbour_buffer=1): # unit = m
    points_pd = points_pd.copy().reset_index()
    bbox_pd = bbox_pd.copy().reset_index()
    
    # find points no bbox
    touch_bool = []
    length = len(points_pd)

    crt_progress = 0
    for idx, row in points_pd.iterrows():

        x_min = row.x_geo - neighbour_buffer
        x_max = row.x_geo + neighbour_buffer
        y_min = row.y_geo - neighbour_buffer
        y_max = row.y_geo + neighbour_buffer

        bbox_near = bbox_pd[(bbox_pd.x0_geo > x_min) & \
                            (bbox_pd.x1_geo < x_max) & \
                            (bbox_pd.y0_geo > y_min) & \
                            (bbox_pd.y1_geo < y_max)]

        if len(bbox_near) == 0:
            touch_bool.append(False)
            continue

        neighbour_bbox = []
        for jdx, jow in bbox_near.iterrows():
            neighbour_bbox.append(shapely.geometry.Polygon([(jow.x0_geo, jow.y0_geo),
                                                            (jow.x1_geo, jow.y0_geo),
                                                            (jow.x1_geo, jow.y1_geo),
                                                            (jow.x0_geo, jow.y1_geo),
                                                            (jow.x0_geo, jow.y0_geo)]))

        neighbour_bbox_multi = shapely.geometry.MultiPolygon(neighbour_bbox)

        circle = shapely.geometry.Point(row.x_geo, row.y_geo).buffer(points_buffer)
        touch_bool.append(neighbour_bbox_multi.intersects(circle))

        pct = round(idx / length * 100)
        if pct > crt_progress:
            print(f"{len(bbox_near)} near bbox find | {idx}/{length} | {crt_progress} %", end='\r')
            crt_progress = pct

    points_pd.loc[:, 'touch'] = touch_bool
    
    # find bbox no points
    touch_bool = []
    length = len(bbox_pd)

    crt_progress = 0
    for idx, row in bbox_pd.iterrows():

        x_min = row.xc_geo - neighbour_buffer
        x_max = row.xc_geo + neighbour_buffer
        y_min = row.yc_geo - neighbour_buffer
        y_max = row.yc_geo + neighbour_buffer

        points_near = points_pd[(points_pd.x_geo > x_min) & \
                                (points_pd.x_geo < x_max) & \
                                (points_pd.y_geo > y_min) & \
                                (points_pd.y_geo < y_max)]

        if len(points_near) == 0:
            touch_bool.append(False)
            continue

        neighbour_points = []
        for jdx, jow in points_near.iterrows():
            neighbour_points.append(shapely.geometry.Point(jow.x_geo, jow.y_geo).buffer(points_buffer))

        neighbour_points_multi = shapely.geometry.MultiPolygon(neighbour_points)

        bbox = shapely.geometry.Polygon([(row.x0_geo, row.y0_geo),
                                         (row.x1_geo, row.y0_geo),
                                         (row.x1_geo, row.y1_geo),
                                         (row.x0_geo, row.y1_geo),
                                         (row.x0_geo, row.y0_geo)])

        touch_bool.append(neighbour_points_multi.intersects(bbox))


        pct = round(idx / length * 100)
        if pct > crt_progress:
            print(f"{len(points_near)} near circle find | {idx}/{length} | {crt_progress} %", end='\r')
            crt_progress = pct

    bbox_pd.loc[:, 'touch'] = touch_bool
    
    return points_pd[~points_pd.touch], bbox_pd[~bbox_pd.touch]


def draw_bbox_points_miss_individual(p4d, ts, points_pd, bbox_pd, point_wrong, bbox_wrong, fig_path, neighbour_buffer=0.5):
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    total_num = len(point_wrong) + len(bbox_wrong)
    crt_num = 0
    # draw points wrong
    for idx, row in point_wrong.iterrows():
        fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=200)

        x_min = row.x_geo - neighbour_buffer
        x_max = row.x_geo + neighbour_buffer
        y_min = row.y_geo - neighbour_buffer
        y_max = row.y_geo + neighbour_buffer

        n_pix = geotiff.geo2pixel(np.asarray([[x_min, y_min],[x_max, y_max]]), geo_head=p4d.dom_header)

        # crop dom out
        x_pix_min, y_pix_min = img_offset = n_pix.min(axis=0)   # also the offset
        w, h = n_pix.max(axis=0) - n_pix.min(axis=0)
        bg = ts.get_crop(page=tifffile.TiffFile(p4d.dom_file).pages[0], i0=y_pix_min, j0=x_pix_min, h=h, w=w)
        ax.imshow(bg)

        # find all dom in this area
        bbox_near = bbox_pd[(bbox_pd.x0_geo > x_min) & \
                            (bbox_pd.x1_geo < x_max) & \
                            (bbox_pd.y0_geo > y_min) & \
                            (bbox_pd.y1_geo < y_max)]

        for jdx, jow in bbox_near.iterrows():
            jbox = np.asarray([(jow.x0_geo, jow.y0_geo),
                               (jow.x1_geo, jow.y0_geo),
                               (jow.x1_geo, jow.y1_geo),
                               (jow.x0_geo, jow.y1_geo),
                               (jow.x0_geo, jow.y0_geo)])
            jbox_pix = geotiff.geo2pixel(jbox, geo_head=p4d.dom_header)
            jbox_pix_off = jbox_pix - img_offset

            ax.plot(*jbox_pix_off.T, 'b-', alpha=0.5)

        # find all near points
        points_near = points_pd[(points_pd.x_geo > x_min) & \
                                (points_pd.x_geo < x_max) & \
                                (points_pd.y_geo > y_min) & \
                                (points_pd.y_geo < y_max)]

        point_near_pix = geotiff.geo2pixel(points_near[['x_geo', 'y_geo']].to_numpy(), 
                                           geo_head=p4d.dom_header) - img_offset
        point_miss_pix = geotiff.geo2pixel(np.asarray([[row.x_geo, row.y_geo]]), 
                                           geo_head=p4d.dom_header) - img_offset

        ax.scatter(*point_near_pix.T, c='r', s=1, alpha=0.3)

        ax.scatter(point_miss_pix[0,0], point_miss_pix[0,1], c='r', s=10, marker='x', alpha=0.7)

        #ax.invert_yaxis()
        ax.axis('off')
        plt.tight_layout()

        plt.savefig(f"{fig_path}/true_nagative_fid{row.fid}_x{point_miss_pix[0,0]+x_pix_min}_y{point_miss_pix[0,1]+y_pix_min}.png")

        plt.clf()
        plt.cla()
        plt.close()
        #del fig, ax

        print(f"idx = {idx} | {crt_num}/{total_num} = {round(crt_num/total_num*100)}%", end='\r')
        crt_num += 1
                
    # draw bbox wrong
    for idx, row in bbox_wrong.iterrows():
        fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)

        x_min = row.xc_geo - neighbour_buffer
        x_max = row.xc_geo + neighbour_buffer
        y_min = row.yc_geo - neighbour_buffer
        y_max = row.yc_geo + neighbour_buffer

        n_pix = geotiff.geo2pixel(np.asarray([[x_min, y_min],[x_max, y_max]]), geo_head=p4d.dom_header)

        # crop dom out
        x_pix_min, y_pix_min = img_offset = n_pix.min(axis=0)   # also the offset
        w, h = n_pix.max(axis=0) - n_pix.min(axis=0)
        bg = ts.get_crop(page=tifffile.TiffFile(p4d.dom_file).pages[0], i0=y_pix_min, j0=x_pix_min, h=h, w=w)
        ax.imshow(bg)

        # find all dom in this area
        bbox_near = bbox_pd[(bbox_pd.x0_geo > x_min) & \
                            (bbox_pd.x1_geo < x_max) & \
                            (bbox_pd.y0_geo > y_min) & \
                            (bbox_pd.y1_geo < y_max)]

        for jdx, jow in bbox_near.iterrows():
            jbox_pix = np.asarray([(jow.x0_dom, jow.y0_dom),
                                   (jow.x1_dom, jow.y0_dom),
                                   (jow.x1_dom, jow.y1_dom),
                                   (jow.x0_dom, jow.y1_dom),
                                   (jow.x0_dom, jow.y0_dom)])
            jbox_pix_off = jbox_pix - img_offset

            ax.plot(*jbox_pix_off.T, 'b--', alpha=0.5)

        ibox_pix_off = np.asarray([(row.x0_dom, row.y0_dom),
                                   (row.x1_dom, row.y0_dom),
                                   (row.x1_dom, row.y1_dom),
                                   (row.x0_dom, row.y1_dom),
                                   (row.x0_dom, row.y0_dom)]) - img_offset

        ax.plot(*ibox_pix_off.T, 'r-', alpha=0.8)

        # find all near points
        points_near = points_pd[(points_pd.x_geo > x_min) & \
                                (points_pd.x_geo < x_max) & \
                                (points_pd.y_geo > y_min) & \
                                (points_pd.y_geo < y_max)]

        point_near_pix = geotiff.geo2pixel(points_near[['x_geo', 'y_geo']].to_numpy(), 
                                           geo_head=p4d.dom_header) - img_offset

        ax.scatter(*point_near_pix.T, c='r', s=1, alpha=0.3)

        ax.axis('off')
        plt.tight_layout()

        plt.savefig(f"{fig_path}/false_positive_bid{idx}_x{row.x0_dom}_y{row.y0_dom}.png")

        plt.clf()
        plt.cla()
        plt.close()
        #del fig, ax

        print(f"idx = {idx} | {crt_num}/{total_num} = {round(crt_num/total_num*100)}%", end='\r')
        crt_num += 1 