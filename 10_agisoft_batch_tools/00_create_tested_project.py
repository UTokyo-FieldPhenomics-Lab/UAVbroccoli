# run in metashape software
# please create metashape project first
import os
import Metashape
from utils import Config

config = Config(json_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json"))

flight_folder = os.listdir(config.img_path)

doc = Metashape.app.document

t1 = Metashape.Tasks.DetectMarkers()
t1.target_type=Metashape.TargetType(5) # CircularTarget16bit

for flight in flight_folder:
    # update chunk lists
    chunk_name_list = [c.label for c in doc.chunks]

    # this need update with _after and _abefore when destructive sampling
    flight_date = flight.split('_')[3]
    flight_prefix = "_0"
    chunk_name = flight_date + flight_prefix

    # examine if already exists:
    if chunk_name in chunk_name_list:
        print(f"[{chunk_name}] already exists")
        continue

    if chunk_name in config.skip_folder:
        print(f"[{chunk_name}] in skip flights")
        continue

    # Not exist, add new chunk:
    print(f"<--------- Adding chunk {chunk_name} --------->")
    chunk = doc.addChunk()
    chunk.label = chunk_name

    # add images:
    print(f"<--------- Adding images --------->")
    ## filter images
    img_list = []
    all_subfile = os.listdir(os.path.join(config.img_path, flight))
    for s in all_subfile:
        if s.endswith('.JPG'):
            img_list.append(os.path.join(config.img_path, flight, s))

    chunk.addPhotos(img_list)

    # detect markers
    print(f"<--------- detecting targets --------->")
    t1.apply(chunk)

    for marker in chunk.markers:
        if "target" in marker.label:  # remove 'target_'
            marker_id = str(int(marker.label[7:]))   # target x
            marker.label = marker_id
        else:
            marker_id = marker.label

    # add gcp
    print(f"<--------- Adding GCP coordinates --------->")
    chunk.importReference(config.ref_csv, 
                          format=Metashape.ReferenceFormat(3), 
                          columns="nxyz", delimiter=",", create_markers=True)

    doc.save()