from config import *

pickle_file = "__pycache__/points_np2d.pkl"
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as handle:
        points_np2d = pickle.load(handle)
else:
    points_np2d = np.zeros((0,2))


for key, val in process_date.items():
    print("\n\n" + "="*50)
    ms = idp.Metashape(val['project'], val['chunk_id'])
    ms.dom = idp.GeoTiff(val['dom'])
    ms.dsm = idp.GeoTiff(val['dsm'])

    print(f"[Loop Start] Load Metashape Project [{ms.project_name}-{ms.chunk_id}]")

    ##################################################
    # backward broccoli root positions to raw images
    ##################################################
    print('-'*50)
    if len(points_np2d) == 0:
        print(f"[Step 1.1] Load head position, this may take several mintues")
        # read the 2D broccoli position
        points = idp.shp.read_shp(root_shp, shp_proj=ms.dom.crs, name_field=0)

        for k, p in points.items():
            points_np2d = np.vstack([points_np2d, p])

        # save to cache, for fast loading next time:
        with open(pickle_file, 'wb') as handle:
            pickle.dump(points_np2d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"[Step 1.1] Use cached head position")

    # read the height of each position
    print('-'*50)
    print(f"[Step 1.2] Read Z values for head positions, this may take several minutes")
    p = idp.ROI()
    p[0] = points_np2d
    p.crs = ms.dom.crs

    # for each point, buffer by 10cm radius and calculate the mean height of this circle
    p.get_z_from_dsm(ms.dsm,  mode='point', buffer=0.1, kernel='mean')

    print('-'*50)
    print(f"[Step 1.3] Backward head positions on raw UAV images")
    out = ms.back2raw(p, ignore='as_point')

    # deeplab_dict = {}
    # for img in out[0].keys():
    #     deeplab_dict[img] = {
    #         "imagePath": ms.photos[img].path,
    #         "points": out[0][img].tolist()
    #     }

    ###############################
    # backward grids to raw images
    ###############################

    # read grid shapefile
    print('-'*50)
    print("[Step 2.1] Load Grid shapefiles")
    roi = idp.ROI(grid_shp, name_field=0)
    roi.get_z_from_dsm(ms.dsm)

    # backward to raw images
    print('-'*50)
    print("[Step 2.2] Backward grid positions on raw UAV images")
    img_dict = ms.back2raw(roi)
    img_dict_sort = ms.sort_img_by_distance(img_dict, roi, num=1)

    # create save folder
    image_save_folder = f"{project_data_folder}/{working_spacename}/{ms.project_name}-{ms.chunk_id}"
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    info_json = {}
    # loops to save images
    for plot_id, val in tqdm(img_dict_sort.items(), "[Step 3] Crop and save images"):
        img_name = list(val.keys())[0]
        coord = list(val.values())[0]

        xmin, ymin = coord.min(axis=0)
        xmax, ymax = coord.max(axis=0)
        xlen, ylen = xmax-xmin, ymax-ymin
        xctr, yctr = (xmax+xmin)/2, (ymax+ymin)/2

        x0, x1 = xctr-crop_image_size/2, xctr+crop_image_size/2
        y0, y1 = yctr-crop_image_size/2, yctr+crop_image_size/2

        poly = np.asarray([
            [x0, y0], [x0, y1], [x1,y1], [x1, y0], [x0, y0]
        ])

        if xlen > crop_image_size or ylen > crop_image_size:
            raise ValueError(
                f"[Warning]: plot [{plot_id}] on img [{img_name}], roi size ({xlen},{ylen}) "
                f"exceed 'crop_image_size' ({crop_image_size}, {crop_image_size})")
        
        imarray = plt.imread(ms.photos[img_name].path)
        cropped, offsets = idp.cvtools.imarray_crop(imarray, poly)

        plt.imsave(f"{image_save_folder}/{plot_id}_{img_name}.png", cropped)

        # calculate the head positions on cropped image
        head_positions = out[0][img_name]
        ## in the cropped range
        head_inside = head_positions[(head_positions[:,0] > xmin) & (head_positions[:,0] < xmax) & 
                                    (head_positions[:,1] > ymin) & (head_positions[:,1] < ymax), :]

        info_json[f'{plot_id}_{img_name}'] = {
            "rawImagePath": ms.photos[img_name].path,
            "cropedImagePath": f"{image_save_folder}/{plot_id}_{img_name}.png",

            "cropLeftTopCorner": offsets,
            
            "gridCoordOnCroppedImage": coord - offsets,

            "headCoordOnRawImage": head_positions,
            "headCoordOnCroppedImage": head_inside - offsets,
        }

    print('-'*50)
    print("[Step 4] Save metadata")
    idp.jsonfile.dict2json(info_json, f"{project_data_folder}/{working_spacename}/{ms.project_name}-{ms.chunk_id}.json")

    