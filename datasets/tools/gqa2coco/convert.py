"""
data_path : path to original GQA annotations to be downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html
img_path : path to original GQA images to be downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html
sg_path : path to original GQA scene graphs to be downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html
vg_img_data_path : path to image info for VG images to be downloaded from https://visualgenome.org/static/data/dataset/image_data.json.zip
"""
import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List
import sys
from tqdm import tqdm

from detectron2.data.detection_utils import read_image

PACKAGE_PARENT = "."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils.spans import consolidate_spans, get_canonical_spans


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the gqa dataset",
    )
    parser.add_argument(
        "--img_path",
        required=True,
        type=str,
        help="Path to the gqa image dataset",
    )
    parser.add_argument(
        "--sg_path",
        required=True,
        type=str,
        help="Path to the gqa dataset scene graph",
    )

    parser.add_argument(
        "--vg_img_data_path",
        required=True,
        type=str,
        help="Path to image meta data for VG"
    )

    parser.add_argument(
        "--out_path",
        required=True,
        default="",
        type=str,
        help="Path where to export the resulting dataset. ",
    )
    return parser.parse_args()


def convert(split, data_path, img_path, sg_path, output_path, imid2data, next_img_id=1, next_id=1):

    print("Loading", data_path / f"{split}_balanced_questions.json")
    with open(data_path / f"{split}_balanced_questions.json", "r") as f:
        data = json.load(f)
    print("Loading", sg_path / f"{split}_sceneGraphs.json")
    with open(sg_path / f"{split}_sceneGraphs.json", "r") as f:
        sg_data = json.load(f)

    img2ann = defaultdict(dict)
    for k, v in data.items():
        img2ann[v["imageId"]][k] = v
    print(len(img2ann))
    print(img2ann["2354786"])
    print(img2ann[list(img2ann.keys())[0]].keys())

    # Add missing annotations by inspecting the semantic field
    regexp = re.compile(r"([0-9]+)")
    regexp2 = re.compile(r"([A-z]+)")
    count = 0

    for k, v in img2ann.items():
        for ann_id, annotations in v.items():
            expected_boxes = []
            for item in annotations["semantic"]:
                if item["operation"] == "select":
                    if len(regexp.findall(item["argument"])) > 0:
                        expected_boxes.append(
                            (regexp2.findall(item["argument"])[0].strip(), regexp.findall(item["argument"])[0])
                        )
            question_boxes = list(annotations["annotations"]["question"].values())

            for name, box_id in expected_boxes:
                if box_id not in question_boxes:
                    count += 1
                    beg = annotations["question"].find(name)
                    end = beg + len(name)
                    annotations["annotations"]["question"][(beg, end)] = box_id

    print(len(img2ann))
    print(img2ann["2354786"])
    print(img2ann[list(img2ann.keys())[0]].keys())
    
    # Add annotations for the questions where there is a box for the answer but not for the question (what/where/who questions)
    for k, v in img2ann.items():
        for ann_id, ann in v.items():
            question_objects = list(ann["annotations"]["question"].values())
            answer_objects = list(ann["annotations"]["answer"].values())
            if len(set(answer_objects) - set(question_objects)) > 0:

                for box_id in answer_objects:
                    if box_id not in question_objects:

                        if ann["question"].find("What") > -1:
                            beg = ann["question"].find("What")
                            end = beg + len("What")
                        elif ann["question"].find("what") > -1:
                            beg = ann["question"].find("what")
                            end = beg + len("what")
                        elif ann["question"].find("Who") > -1:
                            beg = ann["question"].find("Who")
                            end = beg + len("Who")
                        elif ann["question"].find("who") > -1:
                            beg = ann["question"].find("who")
                            end = beg + len("who")
                        elif ann["question"].find("Where") > -1:
                            beg = ann["question"].find("Where")
                            end = beg + len("Where")
                        elif ann["question"].find("where") > -1:
                            beg = ann["question"].find("where")
                            end = beg + len("where")
                        else:
                            continue

                        ann["annotations"]["question"][(beg, end)] = box_id

    print(f"Dumping {split}...")
    # next_img_id = 0
    # next_id = 0

    annotations = []
    images = []

    for k, v in tqdm(img2ann.items()):
        filename = f"{k}.jpg"
        cur_img = {
            "file_name": filename,
            "height": imid2data[int(k)]["height"],
            "width": imid2data[int(k)]["width"],
            "id": next_img_id,
            "original_id": k,
        }

        image = read_image(data_path / "images" / filename, format="BGR")
        if image.shape[1] != cur_img["width"] or image.shape[0] != cur_img["height"]:
            print("before exif correction: ", cur_img)
            cur_img["width"], cur_img["height"] = image.shape[1], image.shape[0]
            print("after exif correction: ", cur_img)
        if filename == "860.jpg":
            print(v)

        for ann_id, annotation in v.items():
            question = annotation["question"]
            answer = annotation["answer"]
            full_answer = annotation["fullAnswer"]

            if len(annotation["annotations"]["question"]) > 0:

                # assert len(annotation["annotations"]["question"]) == 1
                # if len(annotation["annotations"]["question"]) > 1:
                #     print(annotation)
                phrase_all = []
                for text_tok_id, box_anno_id in annotation["annotations"]["question"].items():
                    target_bbox = sg_data[k]["objects"][box_anno_id]
                    x, y, h, w = target_bbox["x"], target_bbox["y"], target_bbox["h"], target_bbox["w"]
                    target_bbox = [x, y, w, h]

                    if isinstance(text_tok_id, str):
                        if ":" in text_tok_id:
                            text_tok_id = text_tok_id.split(":")
                        if isinstance(text_tok_id, list) and len(text_tok_id) > 1:
                            beg = sum([len(x) for x in question.split()[: int(text_tok_id[0])]]) + int(text_tok_id[0])
                            end = (
                                sum([len(x) for x in question.split()[: int(text_tok_id[1]) - 1]])
                                + int(text_tok_id[1])
                                - 1
                            )
                            end = end + len(question.split()[int(text_tok_id[1]) - 1])
                        else:
                            beg = sum([len(x) for x in question.split()[: int(text_tok_id)]]) + int(text_tok_id)
                            end = beg + len(question.split()[int(text_tok_id)])
                    else:
                        beg, end = text_tok_id

                    cleaned_span = consolidate_spans([(beg, end)], question)

                    question_positive = " ".join([question[sp[0]:sp[1]] for sp in cleaned_span])

                    if question_positive.lower() in ["what", "who", "where"]:
                        phrase = answer
                    else:
                        phrase = question_positive
                    phrase_all.append(phrase)

                for text_tok_id, box_anno_id in annotation["annotations"]["question"].items():
                    target_bbox = sg_data[k]["objects"][box_anno_id]
                    x, y, h, w = target_bbox["x"], target_bbox["y"], target_bbox["h"], target_bbox["w"]
                    target_bbox = [x, y, w, h]

                    if isinstance(text_tok_id, str):
                        if ":" in text_tok_id:
                            text_tok_id = text_tok_id.split(":")
                        if isinstance(text_tok_id, list) and len(text_tok_id) > 1:
                            beg = sum([len(x) for x in question.split()[: int(text_tok_id[0])]]) + int(text_tok_id[0])
                            end = (
                                sum([len(x) for x in question.split()[: int(text_tok_id[1]) - 1]])
                                + int(text_tok_id[1])
                                - 1
                            )
                            end = end + len(question.split()[int(text_tok_id[1]) - 1])
                        else:
                            beg = sum([len(x) for x in question.split()[: int(text_tok_id)]]) + int(text_tok_id)
                            end = beg + len(question.split()[int(text_tok_id)])
                    else:
                        beg, end = text_tok_id

                    cleaned_span = consolidate_spans([(beg, end)], question)

                    question_positive = " ".join([question[sp[0]:sp[1]] for sp in cleaned_span])

                    phrase = question_positive
                    if any([phrase.lower().startswith(p) for p in ["what", "who", "where"]]):
                        phrase = answer
                    elif question_positive.lower() == "wh":
                        phrase = answer
                    elif question_positive.lower() == "ho":
                        phrase = answer

                    if sum([1 if p in full_answer else 0 for p in phrase_all]) == 1:
                        if answer in full_answer and phrase in full_answer:
                            phrase = full_answer
                            # beg = full_answer.index(phrase)
                            # end = beg + len(phrase)
                            # print([[(beg, end)]], full_answer, phrase)
                            # cleaned_span, phrase = get_canonical_spans([[(beg, end)]], full_answer)
                            # print(cleaned_span, phrase)

                    if phrase.lower() == "he":
                        if "man" in full_answer or "boy" in full_answer or "guy" in full_answer:
                            phrase = full_answer
                        else:
                            phrase = "man"
                    if phrase.lower() == "she":
                        if "woman" in full_answer or "lady" in full_answer or "girl" in full_answer:
                            phrase = full_answer
                        else:
                            phrase = "woman"

                    if len(phrase) == 2 and not (phrase.lower() == "tv" or phrase.lower() == "cd"):
                        phrase = full_answer

                    if len(phrase) == 1:
                        phrase = full_answer

                    if phrase.lower().startswith("no, "):
                        phrase = phrase[4:]
                    if phrase.lower().startswith("yes, "):
                        phrase = phrase[5:]


                    cur_obj = {
                        "area": h * w,
                        "iscrowd": 0,
                        "category_id": 1,
                        "bbox": target_bbox,
                        "image_id": next_img_id,
                        "id": next_id,
                        "question": question,
                        "answer": answer,
                        "full_answer": full_answer,
                        "tokens_positive": cleaned_span,
                        "question_positive": question_positive,
                        "phrase": phrase,
                    }

                    next_id += 1
                    annotations.append(cur_obj)

        next_img_id += 1
        images.append(cur_img)

    print("images", len(images))
    print("annotations", len(annotations))
    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
    with open(output_path / f"gqa_region_{split}.json", "w") as j_file:
        json.dump(ds, j_file)

    return ds, next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    sg_path = Path(args.sg_path)
    output_path = Path(args.out_path) if args.out_path is not None else data_path

    print("Loading", f"{args.vg_img_data_path}/image_data.json")
    with open(f"{args.vg_img_data_path}/image_data.json", "r") as f:
        image_data = json.load(f)
    imid2data = {x["image_id"]: x for x in image_data}

    os.makedirs(str(output_path), exist_ok=True)

    ds_train, next_img_id, next_id = convert("train", data_path, args.img_path, sg_path, output_path, imid2data)
    ds_val, _, _ = convert("val", data_path, args.img_path, sg_path, output_path, imid2data, next_img_id, next_id)

    ds = {"info": [], "licenses": [], "images": ds_train["images"] + ds_val["images"], "annotations": ds_train["annotations"] + ds_val["annotations"], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
    with open(output_path / f"gqa_region.json", "w") as j_file:
        json.dump(ds, j_file)


if __name__ == "__main__":
    main(parse_args())
