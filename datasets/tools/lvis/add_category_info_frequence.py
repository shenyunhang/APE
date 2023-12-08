import argparse
import copy
import json

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="json_datasets/lvis/lvis_v1_train.json")
    parser.add_argument("--add_freq", action="store_true")
    parser.add_argument("--r_thresh", type=int, default=10)
    parser.add_argument("--c_thresh", type=int, default=100)
    args = parser.parse_args()

    print("Loading", args.json_path)
    json_data = json.load(open(args.json_path, "r"))

    categories = copy.deepcopy(json_data["categories"])

    image_count = {x["id"]: set() for x in categories}
    instance_count = {x["id"]: 0 for x in categories}
    for x in tqdm(json_data["annotations"]):
        if "category_id" in x and x["category_id"] in image_count:
            image_count[x["category_id"]].add(x["image_id"])
            instance_count[x["category_id"]] += 1

    num_freqs = {x: 0 for x in ["r", "f", "c"]}
    for x in categories:
        x["image_count"] = len(image_count[x["id"]])
        x["instance_count"] = instance_count[x["id"]]
        if args.add_freq:
            freq = "f"
            if x["image_count"] < args.c_thresh:
                freq = "c"
            if x["image_count"] < args.r_thresh:
                freq = "r"
            x["frequency"] = freq
            num_freqs[freq] += 1

    for c1, c2 in zip(json_data["categories"], categories):
        print(c1, c2)

    if args.add_freq:
        for x in ["r", "c", "f"]:
            print(x, num_freqs[x])

    cat_info_path = args.json_path[:-5] + "_cat_info.json"
    print("Saving to", cat_info_path)
    json.dump(categories, open(cat_info_path, "w"))

    json_data["categories"] = categories
    print("Saving to", args.json_path)
    json.dump(json_data, open(args.json_path, "w"))
