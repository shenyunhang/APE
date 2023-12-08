import argparse
import json
import os
from collections import defaultdict

import imagesize
from tqdm import tqdm

import utils


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert Open Images annotations into MS Coco format"
    )
    parser.add_argument(
        "-p",
        "--predictions",
        dest="predictions",
        help="coco style prediction file (.json)",
        type=str,
    )
    parser.add_argument(
        "-i", "--image_dir", dest="image_dir", default=None, help="path to images", type=str
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["train", "validation", "test-rvc2020"],
        help="subsets to convert",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="bbox",
        choices=["bbox"],
        help="type of annotations (only bbox supported for now)",
    )
    args = parser.parse_args()
    return args


args = parse_args()

assert (
    args.subset or args.image_dir
), "provide either a split to get image sized from data/ or the directory where the images are stored"

print("loading predictions")
predictions = json.load(open(args.predictions))
print("loading predictions ... Done")

if args.subset:
    image_size_sourcefile = "data/{}_sizes-00000-of-00001.csv".format(args.subset)
    original_image_sizes = utils.csvread(image_size_sourcefile)
    image_size_dict = {x[0]: [int(x[1]), int(x[2])] for x in original_image_sizes[1:]}
    image_ids = list(image_size_dict.keys())
    image_ids.sort()
else:
    image_size_dict = {}
    images = os.listdir(args.image_dir)
    image_ids = [os.path.splitext(x)[0] for x in images]

# prepare per instance information
img_pred_map = defaultdict(list)
for pred in tqdm(predictions, desc="Converting predictions "):
    image_id = pred["image_id"]
    cat = pred["category_id"]

    # Extract height and width
    image_size = image_size_dict.get(image_id, None)
    if image_size is not None:
        image_width, image_height = image_size
    else:
        filename = os.path.join(args.image_dir, image_id + ".jpg")
        image_width, image_height = imagesize.get(filename)
        image_size_dict[image_id] = image_width, image_height

    xmin = pred["bbox"][0] / image_width
    ymin = pred["bbox"][1] / image_height
    xmax = xmin + pred["bbox"][2] / image_width
    ymax = ymin + pred["bbox"][3] / image_height

    conf = pred["score"]

    img_pred_map[image_id].append(f"{cat} {conf:.4f} {xmin:.4f} {ymin:.4f} {xmax:.4f} {ymax:.4f}")


# collect into per image strings
converted_predictions = [["ImageId", "PredictionString"]]
for image_id in image_ids:
    results = img_pred_map[image_id]
    result_string = ""
    for result in results:
        result_string += " " + result

    converted_predictions.append([image_id, result_string[1:]])


outfile = os.path.splitext(args.predictions)[0] + ".csv"
print(f"savind converted predictions to {outfile}")
utils.csvwrite(converted_predictions, outfile)
print(f"savind converted predictions to {outfile} ... Done")
