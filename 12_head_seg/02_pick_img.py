from config import *

import json
from pred import load_model, pred_whole_image
from utils.tolabelme import save_lbme_json

############################
# perpare folders and files
############################

project_data_folder_path = Path(project_data_folder)
annotation_path = Path(annotation_path)
working_space_path = project_data_folder_path.joinpath(working_spacename)

annotation_path.mkdir(exist_ok=True)
print('project_data_folder_path: ', project_data_folder_path)
print('working_space_path: ', working_space_path)
print('annotation_path: ', annotation_path)

######################################################
# create database for recording labeled training data
######################################################

if not os.path.exists(labeled_image_database):
    # json file generation if not have
    file_name_list = []
    resolve_path_list = []
    relative_path_list = []
    date_list = []
    suffix_list = []

    for file in working_space_path.rglob("*"):
        if file.suffix.lower() in supported_suffix:
            # print(os.path.relpath(file, annotation_path))
            suffix_list.append(file.suffix)
            file_name_list.append(file.stem)
            resolve_path_list.append(file.resolve())
            relative_path_list.append(os.path.relpath(file, annotation_path))
            date_list.append(file.parent.stem.split('-')[-1])

    status_list = ['blank'] * len(date_list)

    working_space_dict = {'date': date_list, 
                        'file_name': file_name_list, 
                        'resolve_path': resolve_path_list,
                        'relative_path': relative_path_list,
                        'status': status_list}
    working_space_df = pd.DataFrame(working_space_dict, index=None)

    working_space_df.to_csv(labeled_image_database, index=False)

else:
    working_space_df = pd.read_csv(labeled_image_database, index_col=None)

#####################################################
# random pick some number of images as to-be-labeled
#####################################################

# random select n samples from data
filtered = working_space_df[working_space_df['status']=='blank']

grouped = filtered.groupby('date')
# random_seleted_df = pd.concat([group.sample(n=select_number_per_date, random_state=1) for _, group in grouped])
random_seleted_df = pd.concat([group.sample(n=select_number_per_date) for _, group in grouped])

# check if the `tbd` folder is empty or not
folder_files = [i.name for i in os.scandir(annotation_path)]
if len(folder_files) != 0:
    raise FileExistsError(f"The target folder \n[{annotation_path}] \nis not empty, "
                          f"please manually move the exist {len(folder_files)} files to \n[{annotation_train_path}]")

# not have a trained model, empty project -> create empty json label file:
if not os.path.exists(model_weight):
    ## create empty json file
    for idx, item in random_seleted_df.iterrows():
        print(item.relative_path)
        labelme = {
            'version': "5.0.1",
            'flags': {},
            'shapes': [],
            'imagePath': item.relative_path,
            'imageData': None,
            'imageHeight': crop_image_size,
            'imageWidth': crop_image_size
        }

        # create json file and save it to ./annotations/tbd
        # labelme_json_name = f'tbd_v0_{item.date}_{item.file_name}.json'
        labelme_json_name = f'{item.file_name}.json'
        tbd_label_jsonfile_path = annotation_path.joinpath(labelme_json_name)
    
        with tbd_label_jsonfile_path.open('w+') as f:
            f.write(json.dumps(labelme, indent=1))
# have the pretrained model, run the model and save the model outputs
else:
    # get the index -> json 
    ## walk all *.json files
    json_lists = []
    for f in os.scandir(f"{project_data_folder}/{working_spacename}"):
        if '.json' in f.name:
            json_lists.append(f)

    # get the index of dates
    date_json_link = {}
    for v in pd.unique(random_seleted_df.date):
        for f in json_lists:
            if v in f.name:
                date_json_link[v] = idp.jsonfile.read_json(f.path)

    model = load_model()
    for idx, item in random_seleted_df.iterrows():
        print(item.relative_path)

        info_map = date_json_link[item.date]
        one_img_json = info_map[item.file_name]

        out, whole_mask = pred_whole_image(
            img_path = one_img_json['cropedImagePath'],
            coords   = one_img_json['headCoordOnCroppedImage'],
            model    = model
        )

        labelme_json_name  = f'{item.file_name}.json'
        tbd_label_jsonfile_path = annotation_path.joinpath(labelme_json_name)

        save_lbme_json(one_img_json['cropedImagePath'], out, tbd_label_jsonfile_path)


## change the file status
random_seleted_file_names = random_seleted_df.file_name.to_list()
working_space_df.loc[working_space_df['file_name'].isin(random_seleted_file_names), ['status']] = 'pl' # partially labeled
working_space_df.to_csv(labeled_image_database, index=False)