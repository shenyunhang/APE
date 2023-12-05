import os
import tqdm
from collections import defaultdict
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="datasets/trainval.json")
    parser.add_argument("--image_root", default="datasets/")
    args = parser.parse_args()

    json_path = args.json_path

    print("Loading", json_path)
    with open(json_path, "r") as fr:
        json_data = json.load(fr)

    images = json_data["images"]
    annotations = json_data["annotations"]

    image_id_to_ann = defaultdict(list)
    for ann in annotations:
        image_id_to_ann[ann["image_id"]].append(ann)

    image_id_to_img = defaultdict()
    for img in images:
        image_id_to_img[img["id"]] = img

    """
    {"image": {"image_id": 3032002, "width": 2000, "height": 1500, "file_name": "sa_3032002.jpg"}, "annotations": [{"bbox": [773.0, 789.0, 81.0, 46.0], "area": 2381, "segmentation": {"size": [1500, 2000], "counts": "QT]S18b^13M3N1O2N100O2N100O1O2N1O2N1O100O101N100O10000O100O010O01O1O1N1N3000000001N101N101N2N100O1O1O100O0100O10O0100O10000O01000O100O100O101N2O1N101N2N2O1N2O0O1O2N1N3M3JQ_\\d1"}, "predicted_iou": 0.9558961391448975, "point_coords": [[825.25, 799.1875]], "crop_box": [436.0, 622.0, 692.0, 567.0], "id": 292014182, "stability_score": 0.9862385392189026},],}
    """
    for image_id, ann in tqdm.tqdm(image_id_to_ann.items()):
        img = image_id_to_img[image_id]

        # img["image_id"] = img.pop("id")

        out_data = dict()
        out_data["image"] = img
        out_data["annotations"] = ann

        file_name = img["file_name"]
        file_path = os.path.join(args.image_root, file_name)
        image_ext = file_path.split(".")[-1]
        file_path = file_path[:-len(image_ext)] + "json"

        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as fr:
                    out_data_ = json.load(fr)
                if "key" in out_data_:
                    assert out_data_["key"] == str(out_data["image"]["id"]).zfill(9)
                out_data_.update(out_data)
                out_data = out_data_
            except Exception as e:
                print(e)

        # print("Saving to", file_path)
        # print("out_data", out_data)
        with open(file_path, "w") as fw:
            json.dump(out_data, fw, indent=4)
