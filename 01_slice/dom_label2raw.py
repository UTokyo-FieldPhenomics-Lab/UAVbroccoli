import sys
sys.path.insert(0, f'D:/OneDrive/Program/GitHub/EasyIDP')

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from easyric.caas_lite import TiffSpliter
from easyric.objects import Pix4D
from easyric.io import geotiff

def read_label(project_name, log=False):
    project_path = f"Z:/共享云端硬盘/broccoliProject/tanashi_broccoli2020_RGB_AP/{project_name}"
    raw_img_folder=f"Z:/共享云端硬盘/broccoliProject/rgb_raw/{project_name}"
    label_folder = f"Z:/共享云端硬盘/broccoliProject/pred_bbox_nms_0.3/{project_name}/labels"
    param_folder = f"{project_path}/params"
    
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


if __name__ == "__main__":
    project_name = 'broccoli_tanashi_5_20200528_P4M_10m_before'
    
    keep_bbox, rm_bbox = read_label(project_name, log=True)
    
    print(keep_bbox)
    
    print(rm_bbox)