# Protocol for broccoli data analysis by Metashape

This is the code part of the protocol. The full pipeline including the following steps:

1. Collect and organize the UAV image data
2. 3D reconstruction the field model by Metashape.
3. Broccoli root position detection at early stage by Yolo V5
4. Broccoli head segmetation by BiSeNet V2

The folder structure of this project:

1. `01_slice`, `02_raw_clip`, `03_data_ana` **please ignore them**, they are **previous version** for **Pix4D** project (2019-2021 summer data) only for inner archives, the files are arranged in a mess, hard to use and will not be maintained anymore (but contains draft of deviation)
2. `10_agisoft_batch_tools` is the batch processing code for **Metashape 3D reconstruction**
3. `11_root_pos` and `yolov5` are the code for **root position detection**
4. `12_head_seg` and `bisenet` are the code for **broccoli head segmentation** (**not finished yet**)
5. `EasyIDP` is the code for field map backward projection, while `utils` is the code for image label conversion in previous deep learning steps.

PS: the `yolov5`, `bisenet`, and `easyidp` are submodules for other open source project, after downloading the project, you can download them via git command

```bash
git submodule init
git submodule update
```

## Step 1: Collect UAV data

Please using RTK UAV and automate flight route plan software to control drones to ensure enough overlapping and image quality. Also, please set auto-detectable ground control point board (recommend 75cm x 75 cm for 15m flight) in the field:

| 16bit coded target                                                    | How to get them                                     |
| --------------------------------------------------------------------- | --------------------------------------------------- |
| ![image.png](assets/image-20220115140256-1el73jh.png "16bit coded target") | ![image.png](assets/image-20220120155355-33kjop8.png) |

After collection the images, please organize the data folder as follows:

```plaintxt
.
├── 00_rgb_raw
│   ├── broccoli_tanashi_5_20211021_P4RTK_15m
│   │   ├── DJI_0224.JPG
│   │   ├── ...
│   │   └── DJI_0226.JPG
│   ├── broccoli_tanashi_5_20211025_P4RTK_15m
│   ├── ...
│   ├── broccoli_tanashi_5_20220411_P4RTK_15m
│   └── broccoli_tanashi_5_20220412_P4RTK_15m
├── 01_metashape_projects
│   ├── bbox.pkl
│   ├── broccoli.files
│   ├── broccoli.psx
│   └── outputs
└── 02_GIS
    └── gcp.csv
```

* `00_rgb_raw`: uav image folder, the subfolder is each flight
* `01_metashape_projects`: folder for 3D reconstruction
  * `*.psx `& `*.files` -> metashape project files
  * `outputs` -> produced DOM, DSM maps and point clouds
  * `bbox.pkl` -> plot bounding box files made by our scripts (will be created automatically later)
* `02_GIS`  -> GIS files
  * `gcs.csv` -> ground control points measured by RTK devices, it also need to make the following coded panel first

## Step 2: 3D reconstruction

Please refer to [this document](10_agisoft_batch_tools/readme.md) in `10_agisoft_batch_tools`

## Step 3: Position detection

Please refer to [this document](11_root_pos/readme.md) (not finished) in `11_root_pos`

## Step 4: Head segmentaion

Please refer to [this document](12_head_seg/readme.md) (not finished) in `12_head_seg`
