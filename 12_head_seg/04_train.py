from bisenet import engine
import config
from config import *
from utils import *

from pycocotools.coco import COCO
import json

# convert labelme josn to coco format
annotation_path = Path(annotation_path)
project_data_folder_path = Path(project_data_folder)
working_space_path = project_data_folder_path.joinpath(working_spacename)

if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)
if not os.path.exists(temp_results):
    os.makedirs(temp_results)

@at_path(annotation_path)
def get_coco(annotation_path):
    # print(os.getcwd())
    annotation_path_entry = os.scandir(annotation_path)
    return labelme2coco(annotation_path_entry)

coco_json = get_coco(annotation_path)
with open(config.coco_path, 'w+') as f:
    f.write(json.dumps(coco_json))

coco = COCO(config.coco_path)


# init_bisenet_engine
bisenet_engine = engine.bisenet_engine(config)

# train bisenet
bisenet_engine.train()