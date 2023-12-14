import os
import json
import copy
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks



def compute_area(segmentation):
    if isinstance(segmentation, list):
        polygons = PolygonMasks([segmentation])
        area = polygons.area()[0].item()
    elif isinstance(segmentation, dict):  # RLE
        area = maskUtils.area(segmentation).item()
    else:
        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
    return area

def convert_ref2coco(src_json):
    data = json.load(open(src_json, 'r'))
    inst_idx = 0 # index of the instance
    
    new_data_all = []
    for split in data.keys():
        if split == "train":
            pass
        else:
            continue
        
        new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
        for cur_data in data[split]:
            inst_idx += 1
            image = {"file_name": "COCO_train2014_%012d.jpg"%cur_data["image_id"], "height": cur_data["height"], "width": cur_data["width"], \
                "id": inst_idx,}
            area = compute_area(cur_data["mask"])
            anno = {"bbox":cur_data["bbox"], "segmentation":cur_data["mask"], "image_id":inst_idx, \
                "iscrowd":0, "category_id":1, "id":inst_idx, "area": area, "phrases": cur_data["expressions"]}
            new_data["images"].append(image)
            new_data["annotations"].append(anno)
        assert len(new_data["images"]) == len(data[split])
        assert len(new_data["annotations"]) == len(data[split])

        print(split, {k: len(v) for k, v in new_data.items()})

        new_data_all.append(new_data)
    return new_data_all

if __name__ == "__main__":

    new_data_all = []
    # for dataset in ["refcoco-unc", "refcocog-umd", "refcocoplus-unc", "refcocog-google"]:
    for dataset in ["refcoco-unc", "refcocog-umd", "refcocoplus-unc"]:
        in_json_path = "datasets/SeqTR/%s/instances.json" % dataset
        print(in_json_path)
        new_data = convert_ref2coco(in_json_path)
        new_data_all.extend(new_data)

    file_name_to_annotation = {}
    file_name_to_image = {}

    for new_data in new_data_all:
        images = new_data["images"]
        annotations = new_data["annotations"]

        image_id_to_image = {}
        for image in images:
            image_id_to_image[image["id"]] = image
            if image["file_name"] in file_name_to_image:
                assert file_name_to_image[image["file_name"]]["height"] == image["height"]
                assert file_name_to_image[image["file_name"]]["width"] == image["width"]
                continue
            file_name_to_image[image["file_name"]] = image
        
        for annotation in annotations:
            file_name = image_id_to_image[annotation["image_id"]]["file_name"]
            if file_name in file_name_to_annotation:
                pass
            else:
                file_name_to_annotation[file_name] = []
            file_name_to_annotation[file_name].append(annotation)
    
    images = []
    annotations = []
    cur_image_id = 0
    cur_annotation_id = 0
    for file_name, image in file_name_to_image.items():
        image["id"] = cur_image_id
        images.append(image)

        for annotation in file_name_to_annotation[file_name]:
            phrases = annotation.pop("phrases")
            for phrase in phrases:
                annotation_ = copy.deepcopy(annotation)
                annotation_["phrase"] = phrase
                annotation_["id"] = cur_annotation_id
                annotation_["image_id"] = cur_image_id
                annotations.append(annotation_)
                cur_annotation_id +=1
        
        cur_image_id += 1

    new_data = {"images": images, "annotations": annotations, "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
    print({k: len(v) for k, v in new_data.items()})
    assert len(new_data["images"]) == 28158
    assert len(new_data["annotations"]) == 321327

    # merge train split
    merged_dir = "datasets/SeqTR/refcoco-mixed_group-by-image"
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    merged_json = "datasets/SeqTR/refcoco-mixed_group-by-image/instances_cocofied_train.json"
    
    
    json.dump(new_data, open(merged_json, 'w'))
