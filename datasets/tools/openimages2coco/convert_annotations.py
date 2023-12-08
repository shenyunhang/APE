import argparse
import csv
import json
import os

import utils


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert Open Images annotations into MS Coco format"
    )
    parser.add_argument("-p", "--path", dest="path", help="path to openimages data", type=str)
    parser.add_argument(
        "--version",
        default="v6",
        choices=["v4", "v5", "v6", "challenge_2019"],
        type=str,
        help="Open Images Version",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["val", "train"],
        choices=["train", "val", "test"],
        help="subsets to convert",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="bbox",
        choices=["bbox", "panoptic", "instance"],
        help="type of annotations",
    )
    parser.add_argument(
        "--apply-exif",
        dest="apply_exif",
        action="store_true",
        help="apply the exif orientation correctly",
    )
    parser.add_argument(
        "--exclude-group",
        dest="exclude_group",
        action="store_true",
        help="exclude image and annotation with IsGroupOf=1",
    )
    args = parser.parse_args()
    return args


args = parse_args()
base_dir = args.path
if not isinstance(args.subsets, list):
    args.subsets = [args.subsets]

if args.apply_exif:
    print("-" * 60)
    print("We will apply exif orientation...")
    print("-" * 60)

for subset in args.subsets:
    # Convert annotations
    print("converting {} data".format(subset))

    # Select correct source files for each subset
    if subset == "train" and args.version != "challenge_2019":
        category_sourcefile = "class-descriptions-boxable.csv"
        image_sourcefile = "train-images-boxable-with-rotation.csv"
        if args.version == "v6":
            annotation_sourcefile = "oidv6-train-annotations-bbox.csv"
        else:
            annotation_sourcefile = "train-annotations-bbox.csv"
        image_label_sourcefile = "train-annotations-human-imagelabels-boxable.csv"
        image_size_sourcefile = "train_sizes-00000-of-00001.csv"
        segmentation_sourcefile = "train-annotations-object-segmentation.csv"
        segmentation_folder = "train-masks"

    elif subset == "val" and args.version != "challenge_2019":
        category_sourcefile = "class-descriptions-boxable.csv"
        image_sourcefile = "validation-images-with-rotation.csv"
        annotation_sourcefile = "validation-annotations-bbox.csv"
        image_label_sourcefile = "validation-annotations-human-imagelabels-boxable.csv"
        image_size_sourcefile = "validation_sizes-00000-of-00001.csv"
        segmentation_sourcefile = "validation-annotations-object-segmentation.csv"
        segmentation_folder = "validation-masks"

    elif subset == "test" and args.version != "challenge_2019":
        category_sourcefile = "class-descriptions-boxable.csv"
        image_sourcefile = "test-images-with-rotation.csv"
        annotation_sourcefile = "test-annotations-bbox.csv"
        image_label_sourcefile = "test-annotations-human-imagelabels-boxable.csv"
        image_size_sourcefile = None

    elif subset == "train" and args.version == "challenge_2019":
        category_sourcefile = "challenge-2019-classes-description-500.csv"
        image_sourcefile = "train-images-boxable-with-rotation.csv"
        annotation_sourcefile = "challenge-2019-train-detection-bbox.csv"
        image_label_sourcefile = "challenge-2019-train-detection-human-imagelabels.csv"
        image_size_sourcefile = "train_sizes-00000-of-00001.csv"
        segmentation_sourcefile = "challenge-2019-train-segmentation-masks.csv"
        segmentation_folder = "challenge-2019-train-masks/"

    elif subset == "val" and args.version == "challenge_2019":
        category_sourcefile = "challenge-2019-classes-description-500.csv"
        image_sourcefile = "validation-images-with-rotation.csv"
        annotation_sourcefile = "challenge-2019-validation-detection-bbox.csv"
        image_label_sourcefile = "challenge-2019-validation-detection-human-imagelabels.csv"
        image_size_sourcefile = "validation_sizes-00000-of-00001.csv"
        segmentation_sourcefile = "challenge-2019-validation-segmentation-masks.csv"
        segmentation_folder = "challenge-2019-validation-masks/"

    # Load original annotations
    print("loading original annotations ...", end="\r")
    original_category_info = utils.csvread(
        os.path.join(base_dir, "annotations", category_sourcefile)
    )
    original_image_metadata = utils.csvread(os.path.join(base_dir, "annotations", image_sourcefile))
    original_image_annotations = utils.csvread(
        os.path.join(base_dir, "annotations", image_label_sourcefile)
    )
    original_image_sizes = utils.csvread(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/", image_size_sourcefile)
    )
    if args.task == "bbox":
        original_annotations = utils.csvread(
            os.path.join(base_dir, "annotations", annotation_sourcefile)
        )
    elif args.task == "panoptic" or args.task == "instance":
        original_segmentations = utils.csvread(
            os.path.join(base_dir, "annotations", segmentation_sourcefile)
        )
        original_mask_dir = os.path.join(base_dir, "annotations", segmentation_folder)
        segmentation_out_dir = os.path.join(
            base_dir, "annotations/{}_{}_{}/".format(args.task, subset, args.version)
        )

    print("loading original annotations ... Done")

    oi = {}

    # Add basic dataset info
    print("adding basic dataset info")
    oi["info"] = {
        "contributos": "Vittorio Ferrari, Tom Duerig, Victor Gomes, Ivan Krasin,\
                  David Cai, Neil Alldrin, Ivan Krasinm, Shahab Kamali, Zheyun Feng,\
                  Anurag Batra, Alok Gunjan, Hassan Rom, Alina Kuznetsova, Jasper Uijlings,\
                  Stefan Popov, Matteo Malloci, Sami Abu-El-Haija, Rodrigo Benenson,\
                  Jordi Pont-Tuset, Chen Sun, Kevin Murphy, Jake Walker, Andreas Veit,\
                  Serge Belongie, Abhinav Gupta, Dhyanesh Narayanan, Gal Chechik",
        "description": "Open Images Dataset {}".format(args.version),
        "url": "https://storage.googleapis.com/openimages/web/index.html",
        "version": "{}".format(args.version),
        "year": 2020,
    }

    # Add license information
    print("adding basic license info")
    oi["licenses"] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        },
        {
            "id": 2,
            "name": "Attribution-NonCommercial License",
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
        },
        {
            "id": 3,
            "name": "Attribution-NonCommercial-NoDerivs License",
            "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
        },
        {
            "id": 4,
            "name": "Attribution License",
            "url": "http://creativecommons.org/licenses/by/2.0/",
        },
        {
            "id": 5,
            "name": "Attribution-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-sa/2.0/",
        },
        {
            "id": 6,
            "name": "Attribution-NoDerivs License",
            "url": "http://creativecommons.org/licenses/by-nd/2.0/",
        },
        {
            "id": 7,
            "name": "No known copyright restrictions",
            "url": "http://flickr.com/commons/usage/",
        },
        {
            "id": 8,
            "name": "United States Government Work",
            "url": "http://www.usa.gov/copyright.shtml",
        },
    ]

    # Convert category information
    print("converting category info")
    oi["categories"] = utils.convert_category_annotations(original_category_info)

    # Convert image mnetadata
    print("converting image info ...")
    if subset == "val":
        image_dir = os.path.join(base_dir, "validation")
    else:
        image_dir = os.path.join(base_dir, subset)
    oi["images"] = utils.convert_image_annotations(
        original_image_metadata,
        original_image_annotations,
        original_image_sizes,
        image_dir,
        oi["categories"],
        oi["licenses"],
        args.apply_exif,
    )

    # Convert instance annotations
    print("converting annotations ...")
    # Convert annotations
    if args.task == "bbox":
        oi["annotations"] = utils.convert_instance_annotations(
            original_annotations, oi["images"], oi["categories"], start_index=0
        )

        if args.exclude_group:
            print("=" * 100)
            for k, v in oi.items():
                print(k, len(v))

            IsGroupOf = sum([ann["IsGroupOf"] for ann in oi["annotations"]])
            print("IsGroupOf", IsGroupOf)
            exclude_image_ids = [ann["image_id"] for ann in oi["annotations"] if ann["IsGroupOf"]]
            oi["images"] = [img for img in oi["images"] if img["id"] not in exclude_image_ids]
            oi["annotations"] = [ann for ann in oi["annotations"] if not ann["IsGroupOf"]]

            print("=" * 100)
            for k, v in oi.items():
                print(k, len(v))

    elif args.task == "panoptic":
        oi["annotations"] = utils.convert_segmentation_annotations(
            original_segmentations,
            oi["images"],
            oi["categories"],
            original_mask_dir,
            segmentation_out_dir,
            start_index=0,
        )
        oi["images"] = utils.filter_images(oi["images"], oi["annotations"])
    elif args.task == "instance":
        oi["annotations"] = utils.convert_segmentation_annotations_polygon(
            original_segmentations,
            oi["images"],
            oi["categories"],
            original_mask_dir,
            segmentation_out_dir,
            start_index=0,
        )
        oi["images"] = utils.filter_images(oi["images"], oi["annotations"])

    print("=" * 100)
    for k, v in oi.items():
        print(k, len(v))

    # Write annotations into .json file
    filename = os.path.join(
        base_dir, "annotations/", "openimages_{}_{}_{}.json".format(args.version, subset, args.task)
    )
    if args.exclude_group:
        filename = os.path.join(
            base_dir,
            "annotations/",
            "openimages_{}_{}_{}_nogroup.json".format(args.version, subset, args.task),
        )
    print("writing output to {}".format(filename))
    json.dump(oi, open(filename, "w"))
    print("Done")
