import argparse
import json
from collections import defaultdict

from tqdm import tqdm


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path_box", default="datasets/openimages/annotations/openimages_v6_val_bbox.json"
    )
    parser.add_argument(
        "--json_path_mask",
        default="datasets/openimages/annotations/openimages_v6_val_instance.json",
    )
    parser.add_argument(
        "--json_path_out",
        default="datasets/openimages/annotations/openimages_v6_val_box_instance.json",
    )
    args = parser.parse_args()

    print("Loading", args.json_path_box)
    in_data_box = json.load(open(args.json_path_box, "r"))

    ann_by_image_box = defaultdict(list)
    for ann in tqdm(in_data_box["annotations"]):
        ann_by_image_box[ann["image_id"]].append(ann)

    print("Loading", args.json_path_mask)
    in_data_mask = json.load(open(args.json_path_mask, "r"))

    ann_by_image_mask = defaultdict(list)
    for ann in tqdm(in_data_mask["annotations"]):
        ann_by_image_mask[ann["image_id"]].append(ann)

    ann_by_image_box_key = list(ann_by_image_box.keys())
    ann_by_image_mask_key = list(ann_by_image_mask.keys())
    ann_by_image_key = list(set(ann_by_image_box_key + ann_by_image_mask_key))

    print("ann_by_image_box_key", len(ann_by_image_box_key))
    print("ann_by_image_mask_key", len(ann_by_image_mask_key))
    print("ann_by_image_key", len(ann_by_image_key))

    num_merge_instance = 0
    num_merge_image = 0
    ann_by_image = defaultdict(list)
    for image_id in tqdm(ann_by_image_key):
        if image_id in ann_by_image_box:
            anns_box = ann_by_image_box[image_id]
        else:
            anns_box = []
        if image_id in ann_by_image_mask:
            anns_mask = ann_by_image_mask[image_id]
        else:
            anns_mask = []

        if anns_box == None:
            ann_by_image[image_id] = anns_mask
            continue

        num_merge = 0
        for ann_box in anns_box:
            xywh_box = ann_box["bbox"]
            box_box = {
                "x1": xywh_box[0],
                "y1": xywh_box[1],
                "x2": xywh_box[0] + xywh_box[2],
                "y2": xywh_box[1] + xywh_box[3],
            }
            for ann_mask in anns_mask:
                xywh_mask = ann_mask["bbox"]
                box_mask = {
                    "x1": xywh_mask[0],
                    "y1": xywh_mask[1],
                    "x2": xywh_mask[0] + xywh_mask[2],
                    "y2": xywh_mask[1] + xywh_mask[3],
                }

                iou = get_iou(box_box, box_mask)

                # print(xywh_box, xywh_mask)
                if iou > 0.99 and ann_box["category_id"] == ann_mask["category_id"]:
                    ann_box["segmentation"] = ann_mask["segmentation"]

                    num_merge += 1
                    # print(num_merge)

                    break
            else:
                seg = []
                # left_top
                seg.append(xywh_box[0])
                seg.append(xywh_box[1])
                # left_bottom
                seg.append(xywh_box[0])
                seg.append(xywh_box[1] + xywh_box[3])
                # right_bottom
                seg.append(xywh_box[0] + xywh_box[2])
                seg.append(xywh_box[1] + xywh_box[3])
                # right_top
                seg.append(xywh_box[0] + xywh_box[2])
                seg.append(xywh_box[1])

                ann_box["segmentation"] = [seg]
        ann_by_image[image_id] = anns_box

        if num_merge > 0:
            num_merge_instance += num_merge
            num_merge_image += 1

    annotations = [ann for anns in ann_by_image.values() for ann in anns]
    in_data_box["annotations"] = annotations

    print("num_merge_instance", num_merge_instance)
    print("num_merge_image", num_merge_image)
    print("len(annotations)", len(annotations))

    print("Saving to", args.json_path_out)
    json.dump(in_data_box, open(args.json_path_out, "w"))
