import os
import json

if __name__ == "__main__":
    for dataset in ["refcoco-unc", "refcocog-umd", "refcocoplus-unc", "refcocog-google"]:
        in_json_path = "datasets/SeqTR/%s/instances.json" % dataset
        out_json_path = "datasets/SeqTR/%s/instances_cocofied.json" % dataset
        os.system("python3.9 datasets/tools/seqtr2coco/convert_ref2coco.py --src_json %s --des_json %s" %(in_json_path, out_json_path))

    # merge train split
    merged_dir = "datasets/SeqTR/refcoco-mixed"
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    merged_json = "datasets/SeqTR/refcoco-mixed/instances_cocofied_train.json"
    inst_idx = 0 # index of the instance
    new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
    for dataset in ["refcoco-unc", "refcocog-umd", "refcocoplus-unc"]:
        json_path = "datasets/SeqTR/%s/instances_cocofied_train.json" % dataset
        data = json.load(open(json_path, 'r'))
        # for split in data.keys():
        for (img, anno) in zip(data["images"], data["annotations"]):
            inst_idx += 1
            img["id"] = inst_idx
            anno["image_id"] = inst_idx
            anno["id"] = inst_idx
            new_data["images"].append(img)
            new_data["annotations"].append(anno)
    print({k: len(v) for k, v in new_data.items()})
    assert len(new_data["images"]) == 126908
    assert len(new_data["annotations"]) == 126908
    json.dump(new_data, open(merged_json, 'w')) # 126908 referred objects
