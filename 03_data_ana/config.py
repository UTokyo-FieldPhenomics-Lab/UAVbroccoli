import os
import sys
import platform

user = platform.node()

if user == "DESKTOP-3J8SGRC":
    easyidp_path = "Y:\hwang_Pro\github\EasyIDP"
else:
    raise FileNotFoundError(f"please add new user [{user}] setting in configs.py")
    
sys.path.insert(0, easyidp_path)


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
    
    def __init__(self, key, year=2020):
        
        if year == 2020:
            self.root = r"Y:\hwang_Pro\data\2020_tanashi_broccoli"
        elif year == 2019:
            self.root = r"Y:\hwang_Pro\data\2019_tanashi_broccoli5"
        else:
            self.root = r"Y:\hwang_Pro\data\2021_tanashi_broccoli"
        
        self.project_name = self.generate_name(key)
        
        self.raw_img =f"{self.root}/00_rgb_raw/{self.project_name}"
        if year == 2020:
            self.pix4d_project = f"{self.root}/01_tanashi_broccoli2020_RGB_AP/{self.project_name}"
            self.pix4d_param = f"{self.pix4d_project}/params"
        elif year == 2019:
            self.pix4d_project = f"{self.root}/01_pix4d_projects/{self.project_name}"
            self.pix4d_param = f"{self.pix4d_project}/params"
        else:
            self.pix4d_project = f"{self.root}/01_pix4d_projects/{self.project_name}"
            self.pix4d_param = f"{self.pix4d_project}/1_initial/params"
        
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
                    "0528_p"  :'broccoli_tanashi_5_20200528_P4M_10m_before',
                    # ============== 2021 ==============
                    "210412"  :"broccoli_tanashi_5_20210412_P4RTK_15m_M",
                    "210512"  :"broccoli_tanashi_5_20210512_P4RTK_15m_M",
                    "210514"  :"broccoli_tanashi_5_20210514_P4RTK_15m_M_abefore",
                    "210515"  :"broccoli_tanashi_5_20210515_P4RTK_15m_M",
                    "210519"  :"broccoli_tanashi_5_20210519_P4RTK_15m_M_abefore",
                    "210520"  :"broccoli_tanashi_5_20210520_P4RTK_15m_M_abefore",
                    "210526"  :"broccoli_tanashi_5_20210526_P4RTK_15m_M",
                    # ============== 2019 ==============
                    "191001"  :"broccoli_tanashi_5_20191001_mavicRGB_15m",
                    "191006"  :"broccoli_tanashi_5_20191006_mavicRGB_15m_wind_M",
                    "191008"  :"broccoli_tanashi_5_20191008_mavicRGB_15m_M",
                    "191023"  :"broccoli_tanashi_5_20191023_mavicRGB_15m_M_2",
                    "191106"  :"broccoli_tanashi_5_20191106_mavicRGB_15m_M_2",
                    #"191129"  :"broccoli_tanashi_5_20191129_mavicRGB_15m_after",
                    "191129"  :"broccoli_tanashi_5_20191129_mavicRGB_15m_before",
                    #"191210"  :"broccoli_tanashi_5_20191210_mavicRGB_15m_after",
                    #"191210"  :"broccoli_tanashi_5_20191210_mavicRGB_15m_before",
                    "191210"  :"broccoli_tanashi_5_20191210_mavicRGB_15m_before_2",
                   }
        
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