from config import *

from pred import load_model, pred_whole_image
from utils.tolabelme import save_lbme_json

def show_seg_results(img_path, coords, out, save_path, size=1000, dpi=72):
    coords = np.asarray(coords)

    fig, ax = plt.subplots(1,1, figsize=(size/dpi,size/dpi), dpi=dpi)

    img = plt.imread(img_path)
    ax.imshow(img)

    ax.scatter(*coords.T, facecolors='b', edgecolors='w', linewidths=2, s=70)

    for o in out:
        o = np.asarray(o)
        ax.plot(*o.T, 'r-')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

working_space_df = pd.read_csv(labeled_image_database, index_col=None)

# get the index -> json 
## walk all *.json files
json_lists = []
for f in os.scandir(f"{project_data_folder}/{working_spacename}"):
    if '.json' in f.name:
        json_lists.append(f)

# get the index of dates
date_json_link = {}
for v in pd.unique(working_space_df.date):
    for f in json_lists:
        if v in f.name:
            date_json_link[v] = idp.jsonfile.read_json(f.path)

model = load_model()
out_dict = {}
pbar = tqdm(working_space_df.iterrows(), total=working_space_df.shape[0])
for idx, item in pbar:
    pbar.set_description(f"Processing [{item.date}]-[{item.file_name}]")

    info_map = date_json_link[item.date]
    one_img_json = info_map[item.file_name]

    date_folder = f"{result_folder}/{item.date}"
    if item.date not in out_dict.keys():
        out_dict[item.date] = {}

    if not os.path.exists(date_folder):
        os.makedirs(date_folder)

    out, whole_mask = pred_whole_image(
        img_path = one_img_json['cropedImagePath'],
        coords   = one_img_json['headCoordOnCroppedImage'],
        model    = model
    )

    # draw figures
    show_seg_results(
        img_path  = one_img_json['cropedImagePath'], 
        coords    = one_img_json['headCoordOnCroppedImage'],
        out       = out,
        save_path = f'{date_folder}/{item.file_name}.png',
    )

    save_lbme_json(
        one_img_json['cropedImagePath'], 
        out, 
        f'{date_folder}/{item.file_name}.json')
    
    out_dict[item.date][item.file_name] = {
        'out': out,
        'cropLeftTopCorner': one_img_json['cropLeftTopCorner'],
        'gridCoordOnCroppedImage': one_img_json['gridCoordOnCroppedImage'],
        'headCoordOnCroppedImage': one_img_json['headCoordOnCroppedImage'],
    }

idp.jsonfile.save_json(out_dict, f"{result_folder}/out_results.json")