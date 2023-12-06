import argparse
import json
import os
from collections import defaultdict

import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="datasets/trainval.json")
    parser.add_argument("--json_path_out", default="datasets/trainval.json")
    args = parser.parse_args()

    json_path = args.json_path

    print("Loading", json_path)
    with open(json_path, "r") as fr:
        json_data = json.load(fr)

    print("=" * 100)
    for k, v in json_data.items():
        print(k, len(v))

    images = json_data["images"]
    annotations = json_data["annotations"]

    image_id_to_ann = defaultdict(list)
    for ann in annotations:
        image_id_to_ann[ann["image_id"]].append(ann)

    image_id_to_img = defaultdict()
    for img in images:
        # assert img["id"] not in image_id_to_img.keys()
        image_id_to_img[img["id"]] = img

    new_images = []
    new_annotations = []

    num_group_img = 0
    num_group_ann = 0
    num_group_img_ann = 0
    for image_id, ann in tqdm.tqdm(image_id_to_ann.items()):
        skip_image = False
        for an in ann:
            if an["IsGroupOf"] == 1:
                skip_image = True
                num_group_ann += 1
        if skip_image:
            num_group_img += 1
            num_group_img_ann += len(ann)
            image_id_to_img.pop(image_id)
            # image_id_to_ann.pop(image_id)
            continue

        img = image_id_to_img[image_id]

        new_images.append(img)
        new_annotations.extend(ann)

    # json_data["images"] = new_images
    json_data["images"] = list(image_id_to_img.values())
    json_data["annotations"] = new_annotations

    print("num_group_img", num_group_img)
    print("num_group_ann", num_group_ann)
    print("num_group_img_ann", num_group_img_ann)

    print("=" * 100)
    for k, v in json_data.items():
        print(k, len(v))

    print("Saving to", args.json_path_out)
    with open(args.json_path_out, "w") as fw:
        json.dump(json_data, fw)
