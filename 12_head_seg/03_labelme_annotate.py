from config import *
import subprocess

#################################
# run labelme GUI to label files
#################################
subprocess.check_output(
    f"python ../labelme/__main__.py {annotation_path} --nodata", 
    stderr=subprocess.STDOUT,
    shell=True)