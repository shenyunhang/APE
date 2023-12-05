# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import copy
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", default="datasets/imagenet/annotations/imagenet-21k_image_info.json"
    )
    parser.add_argument("--out_path", default="")
    args = parser.parse_args()

    print("Loading", args.in_path)
    in_data = json.load(open(args.in_path, "r"))

    print(in_data["images"][0])
    for x in in_data["images"]:
        x["neg_category_ids"] = []
        x["not_exhaustive_category_ids"] = []
    print(in_data["images"][0])

    print(in_data["categories"][0])
    for x in in_data["categories"]:
        x["frequency"] = "f"
    print(in_data["categories"][0])

    if args.out_path != "":
        for k, v in in_data.items():
            print("data", k, len(v))
        print("Saving to", args.out_path)
        json.dump(in_data, open(args.out_path, "w"))
