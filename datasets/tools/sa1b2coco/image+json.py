import argparse
import json
import os
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert image list to coco format")
    parser.add_argument("--image_root", help="img root", required=True)
    parser.add_argument("--json_path", help="output path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    imgs = []
    for root, dirs, files in os.walk(args.image_root):
        print(root)

        files = [f for f in files if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"))]
        for file_name in tqdm.tqdm(files):
            path = os.path.join(root, file_name)
            rpath = path.replace(args.image_root, "")

            json_path = path[:-len(path.split(".")[-1])] + "json"

            if os.path.isfile(json_path):
                pass
            else:
                continue

            try:
                json_data = json.load(open(json_path, "r"))
            except Exception as e:
                print(json_path)
                print(e)
                continue

            assert json_data["image"]["file_name"] == file_name, json_data
            image_id = json_data["image"]["image_id"]
            height = json_data["image"]["height"]
            width = json_data["image"]["width"]
            img = {"file_name": rpath, "height": height, "width": width, "id": image_id,}
            
            imgs.append(img)

            if len(imgs) % 1000000 == 0:
                save(imgs, args, suffix="_" + str(len(imgs)))

        print("#imgs", len(imgs))

    save(imgs, args)

def save(imgs, args, suffix=""):
    json_data = dict()
    json_data["categories"] = [{"id": 1, "name": "object"}]
    json_data["images"] = imgs
    json_data["annotations"] = []

    print("json_data", json_data.keys())
    print("categories", len(json_data["categories"]))
    print("#images", len(json_data["images"]))

    with open(args.json_path + suffix + ".json", "w") as f:
        json.dump(json_data, f)

    print(args)


if __name__ == "__main__":
    main()
