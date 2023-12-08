import argparse
import copy
import json

from nltk.corpus import wordnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="datasets/imagenet/annotations/winter21_whole.json")
    args = parser.parse_args()

    print("Loading", args.json_path)
    in_data = json.load(open(args.json_path, "r"))

    categories = copy.deepcopy(in_data["categories"])

    for category in categories:
        wnid = category["wnid"]

        synset = wordnet.synset_from_pos_and_offset("n", int(wnid[1:]))
        synonyms = [x.name() for x in synset.lemmas()]

        category["synset"] = synset.name()
        category["name"] = synonyms[0]
        category["def"] = synset.definition()
        category["synonyms"] = synonyms

    for c1, c2 in zip(in_data["categories"], categories):
        print(c1, c2)

    in_data["categories"] = categories

    # print("Saving to", args.json_path)
    # json.dump(in_data, open(args.json_path, "w"))
