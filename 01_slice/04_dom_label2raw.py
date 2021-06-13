from config import *

project_name = 'broccoli_tanashi_5_20200528_P4M_10m_before'
keep_bbox, rm_bbox = read_label(project_name, log=True)

print(keep_bbox)
print(rm_bbox)