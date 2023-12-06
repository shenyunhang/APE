
import os
import json

from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog

import sota_t


print(MetadataCatalog.keys())

for k, v in MetadataCatalog.items():
    if k.startswith("odinw"):
        pass
    else:
        continue
    if v.get("json_file", None):
        
        pass
    else:
        continue

    print("-" * 100)
    print(k, v.json_file)

    assert "annotations_without_background" in v.json_file

    json_file = v.json_file

    json_dir = os.path.dirname(json_file)
    json_name = os.path.basename(json_file)
    if "_converted.json" in json_name:
        json_file = os.path.join(json_dir, json_name.replace("_converted.json", ".json"))
    else:
        json_file = os.path.join(json_dir, json_name)

    print("Open \t", json_file)
    with open(json_file, "r") as fr:
        json_data = json.load(fr)

    images = json_data["images"]
    annotations = json_data["annotations"]

    image_id = 1
    old_image_id_to_new_image_id = {}
    for img in images:
        assert img["id"] not in old_image_id_to_new_image_id
        old_image_id_to_new_image_id[img["id"]] = image_id
        img["id"] = image_id
        image_id += 1

    annotation_id = 1
    for ann in annotations:
        ann["image_id"] = old_image_id_to_new_image_id[ann["image_id"]]
        ann["id"] = annotation_id
        annotation_id += 1
    
    json_data["images"] = images
    json_data["annotations"] = annotations

    if "_converted.json" not in json_name:
        json_file = os.path.join(json_dir, json_name.replace(".json", "_converted.json"))
    else:
        json_file = os.path.join(json_dir, json_name)
    print("Save \t", json_file)
    with open(json_file, "w") as fw:
        json.dump(json_data, fw, indent=4)





