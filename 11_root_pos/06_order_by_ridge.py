from config import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

# 转换为单位向量
def vector_mod(Ax, By):
    return np.sqrt(np.sum(np.square([Ax, By])))

def transform_coord(process_area_px, center_points_px, ridge_direction):
    if ridge_direction in ["x", "X"]:
        xu = process_area_px[1,:] - process_area_px[0,:]
        yu = process_area_px[1,:] - process_area_px[2,:]
    elif ridge_direction in ["y", "Y"]:
        yu = process_area_px[1,:] - process_area_px[0,:]
        xu = process_area_px[1,:] - process_area_px[2,:]
    else:
        raise ValueError(f"Only x and Y are accepted, not {ridge_direction}")

    v1 = xu / vector_mod(*xu)
    v2 = yu / vector_mod(*yu)

    cvtmat = np.vstack([v1, v2]).T

    cvt_xy = np.linalg.inv(cvtmat).dot(center_points_px.T).T
    process_area_cvt = np.linalg.inv(cvtmat).dot(process_area_px.T).T
    return cvt_xy, process_area_cvt


# read data
header = geotiff.get_header(dom_path_for_detection)
proj = header['proj']
process_area = shp.read_shp2d(field_rectange_bounding_box_shapefile,
                            shp_proj=proj)

center_points = shp.read_shp2d(edited_center_points_shapefile,
                            shp_proj=proj)

center_points_np = np.concatenate(list(center_points.values()),0)
center_points_px = geotiff.geo2pixel(center_points_np, header)

bound_np = list(process_area.values())[0]  # equal to process_area['0'], but the key depend on different files
process_area_px = geotiff.geo2pixel(bound_np, header)

cvt_xy, process_area_cvt = transform_coord(process_area_px, center_points_px, ridge_direction)

check_ridge_onoff = True

while check_ridge_onoff:
    v, n = np.unique((cvt_xy[:,0]/ ridge_strength_ratio).astype(int) * ridge_strength_ratio, return_counts=True)
    peaks, _ = find_peaks(n, distance=ridge_distance_parameters, height=ridge_height_parameters)

    # draw preview figures
    fig, ax = plt.subplots(2,1, figsize=(12,12), gridspec_kw={'height_ratios': [1, 4]})

    ax[0].plot(v,n)
    for i in peaks:
        ax[0].plot(v[i], n[i], 'rx')
        ax[1].axvline(x=v[i],c='r', lw=1)
        
    ax[0].axhline(y=ridge_height_parameters, c='g')
        
    ax[1].plot(*process_area_cvt.T)
    ax[1].scatter(*cvt_xy.T, s=5)

    ax[0].set_ylabel(f"height_thresh={ridge_height_parameters}")
    ax[0].set_xlabel(f"distance_parameter={ridge_distance_parameters}")
    ax[0].set_title(f"strength_ratio={ridge_strength_ratio}, direction={ridge_direction}")

    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{figure_save_path}/ridge_detection_confirm.png", dpi=300) 
    plt.show()

    # check if ridges are correctly detected
    ans = input("\nIs all the ridges are correct?[Y/N]\n>>> ")

    plt.close(fig)

    if ans in ["y", "yes", "Y", "True"]:
        check_ridge_onoff = False
    else:
        ask_str = \
f"""
Please type parameter NUMBER want adjust:
[1] Strength Ratio = {ridge_strength_ratio}
[2] ridge direction = {ridge_direction}
[3] distance parameters = {ridge_distance_parameters}
[4] height parameters = {ridge_height_parameters}
>>> """
        selection_onoff = True
        while selection_onoff:
            selection = int(input(ask_str))
            if selection in [1,2,3,4]:
                selection_onoff = False

        change_value = input(f"Change to?\n>>> ")

        if selection == 1:
            ridge_strength_ratio = float(change_value)
        elif selection == 2:
            ridge_direction = change_value
            cvt_xy, process_area_cvt = transform_coord(process_area_px, center_points_px, ridge_direction)
        elif selection == 3:
            ridge_distance_parameters = float(change_value)
        elif selection == 4:
            ridge_height_parameters = float(change_value)

#####################
# regress by RANSAC #
#####################

## find the ridge proper distance (assume ridge dirstribute equally)
peak_x = v[peaks]
buffer = peak_x[1:] - peak_x[:-1]
buffer = np.hstack([max(buffer), buffer, max(buffer)])

check_ransac_onoff = True

while check_ransac_onoff:
    order_df = pd.DataFrame({"x_geo": center_points_np[:,0], 
                             "y_geo": center_points_np[:,1],
                             "cvt_x": cvt_xy[:,0],
                             "cvt_y": cvt_xy[:,1],
                             "fid": -1})

    fid_st = 1

    ## draw figure & rename in the same loop
    fig, ax = plt.subplots(1,1, figsize=(15,15))

    for i in peaks:
        ax.axvline(x=v[i],c='gray', lw=1, alpha=0.5)

    for i, p in enumerate(peak_x):
        range_st = p - buffer[i] * ridge_buffer
        range_ed = p + buffer[i+1] * ridge_buffer
        
        selected = (order_df.cvt_x >= range_st ) & (order_df.cvt_x <= range_ed)
        selected_xy = order_df.loc[selected, ['cvt_x', 'cvt_y']]

        if use_ransac:
            # here flip x and y, because vertical line residual is very high
            # > residual = base_estimator.predict(X) - y
            Y = np.ones((len(selected_xy), 1))
            Y[:, 0] = selected_xy.cvt_y

            reg = RANSACRegressor(residual_threshold=50, max_trials=1000).fit(Y, selected_xy['cvt_x'])
            inlier_mask = reg.inlier_mask_
        else:
            inlier_mask = np.ones(len(selected_xy)).astype(bool)
        
        # remove outlier and sort by Y axis
        selected_xy_sort = selected_xy.loc[inlier_mask,:].sort_values(by=['cvt_y'], ascending=False)
        fid = np.linspace(fid_st, len(selected_xy_sort)+fid_st-1, num=len(selected_xy_sort)).astype(int)
        
        selected_xy_sort['fid'] = fid
        order_df.loc[selected_xy_sort.index, 'fid'] = selected_xy_sort.fid

        non_selected_xy = selected_xy.loc[~inlier_mask,:]
        
        color=np.random.rand(3)
        ax.scatter(selected_xy_sort.cvt_x, selected_xy_sort.cvt_y, color=color)
        ax.scatter(non_selected_xy.cvt_x, non_selected_xy.cvt_y, color=color, edgecolors='k', alpha=0.3)

        fid_st += len(selected_xy_sort)

    fid_positive = order_df[order_df.fid > 0]
    fid_negative = order_df[order_df.fid < 0]
    total_num = len(order_df)
    marked_num = len(fid_positive)
    ignored_num = len(fid_negative)
    
    if use_ransac:
        ax.set_title(f"Based on ridge buffer={ridge_buffer}, Regressed by RANSAC, nresidual_threshold={ransac_residual_threshold}, max_trails={ransac_max_trials}")
    else:
        ax.set_title(f"Regressed by simple ridge buffer={ridge_buffer}")
    ax.set_xlabel(f"Note: {ignored_num} dots with black boundary are not labeled ({marked_num}/{total_num})")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"{figure_save_path}/ridge_detection_final.png", dpi=300)
    plt.show()

    # check if ridges are correctly detected
    ans = input(f"Is all the ridges colors are correct? {marked_num} of {total_num} re-labeled, {ignored_num} ignored[Y/N]\n>>> ")

    plt.close(fig)

    if ans in ["y", "yes", "Y", "True"]:
        check_ransac_onoff = False
    else:
        ask_str = \
f"""
Please type parameter NUMBER want adjust:
[1] Ridge buffer = {ridge_buffer}
[2] Use RANSAC = {use_ransac}
[3] RANSAC residual threshold = {ransac_residual_threshold}
[4] RANSAC max trails = {ransac_max_trials}
>>> """
        selection_onoff = True
        while selection_onoff:
            selection = int(input(ask_str))
            if selection in [1,2,3, 4]:
                selection_onoff = False

        change_value = input(f"Change to?\n>>> ")
        if selection == 1:
            if float(change_value) >= 0.8 or float(change_value) <= 0:
                print("0 < Ridge buffer < 0.8")
                continue
            else:
                ridge_buffer = float(change_value)
        elif selection == 2:
            if change_value in ["0", "f", "F", "False", "false"]:
                use_ransac = False
            else:
                use_ransac = True
        elif selection == 3:
            ransac_residual_threshold = int(change_value)
        elif selection == 4:
            ransac_max_trials = int(change_value)

# save results to shp file
selection_onoff = True
if ignored_num == 0:
    save_shp(order_df, f"{shapefile_save_folder}/ordered_center_points", type='points')
    selection_onoff = False

while selection_onoff:
    ans = input(f"Successfully labeled {marked_num} in {total_num}, for {ignored_num} non-labeled points\n[1] give continues id\n[2] set id = -1\n[3] remove these points\n>>> ")
    if ans == '1':
        selection_onoff = False
        fid = np.linspace(fid_st, ignored_num+fid_st-1, num=ignored_num).astype(int)
        fid_negative['fid'] = fid
        order_df.loc[fid_negative.index, 'fid'] = fid_negative.fid
        save_shp(order_df, f"{shapefile_save_folder}/ordered_center_points", type='points')
    elif ans == '2':
        selection_onoff = False
        save_shp(order_df, f"{shapefile_save_folder}/ordered_center_points", type='points')
    elif ans == '3':
        selection_onoff = False
        save_shp(fid_positive, f"{shapefile_save_folder}/ordered_center_points", type='points')
    else:
        print(f"Please type 1, 2, 3, not [{ans}]")
    