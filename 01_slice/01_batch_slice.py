from config import *

#project_names = ['0514_p', '0518_p', '0520_p']

project_names = ['0525_p', '0526_p']

for pn in project_names:
    slice_one_plot(pn, format="tif")
