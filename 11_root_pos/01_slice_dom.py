from config import *
import os

dom_name = os.path.split(dom_path_for_detection)[1]
# -> broccoli_tanashi_dom.tif

clip_folder = os.path.join(dom_slice_save_folder, dom_slice_save_format, dom_name)
# -> 10_root_posotion/jpg/broccoli_tanashi_dom.tif/

if not os.path.exists(clip_folder):
    os.makedirs(clip_folder)

json_name = clip_folder + '.json'
# -> 10_root_posotion/jpg/broccoli_tanashi_dom.tif.json

ts = TiffSpliter(tif_path=dom_path_for_detection, grid_h=dom_slice_length, grid_w=dom_slice_length, grid_buffer=dom_slice_buffer)

offset_json = {}
for w_id, w_st in enumerate(ts.wgrid_st):
    for h_id, h_st in enumerate(ts.hgrid_st):
        tiff_name = ts.id2name(w_id=w_id, h_id=h_id)
        offset_json[tiff_name] = {'x':ts.wgrid_st[w_id], 
                                    'y':ts.hgrid_st[h_id]}
ts.dict2json(offset_json, json_name)

ts.save_all_grids(save_folder=clip_folder, extend=True, skip_empty=True, format=dom_slice_save_format)