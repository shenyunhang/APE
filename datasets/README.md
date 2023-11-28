
# Detectron2 Builtin Datasets

Detectron2 has builtin support for a few datasets.

The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.

Under this directory, following [here](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md) to prepare COCO, LVIS, cityscapes, Pascal VOC and ADE20k.

The expected structure is described below.
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
  cityscapes/
  VOC20{07,10,12}/
  ADEChallengeData2016/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


# APE Builtin Datasets

## Expected dataset structure for [Objects365](https://data.baai.ac.cn/details/Objects365_2020):
```
$DETECTRON2_DATASETS/
  objects365/
    annotations/
      objects365_train_fixname.json
      objects365_val_fixname.json
    train/
      images/
    val/
      images/
```

After downloading and extracting Objects365, create a symbolic link to `./datasets/`.
Then, run
```bash
python3 tools/objects3652coco/get_image_info.py --image_dir datasets/objects365/train/ --json_path datasets/objects365/annotations/zhiyuan_objv2_train.json --output_path datasets/objects365/annotations/image_info_train.txt
python3 tools/objects3652coco/get_image_info.py --image_dir datasets/objects365/val/ --json_path datasets/objects365/annotations/zhiyuan_objv2_val.json --output_path datasets/objects365/annotations/image_info_val.txt

python3 tools/objects3652coco/convert_annotations.py --root_dir datasets/objects365/ --image_info_path datasets/objects365/annotations/image_info_train.txt --subsets train --apply_exif
python3 tools/objects3652coco/convert_annotations.py --root_dir datasets/objects365/ --image_info_path datasets/objects365/annotations/image_info_val.txt --subsets val --apply_exif
python3 tools/objects3652coco/convert_annotations.py --root_dir datasets/objects365/ --image_info_path datasets/objects365/annotations/image_info_val.txt --subsets minival --apply_exif

python3 tools/objects3652coco/fix_o365_names.py --ann datasets/objects365/annotations/objects365_train.json
python3 tools/objects3652coco/fix_o365_names.py --ann datasets/objects365/annotations/objects365_val.json
python3 tools/objects3652coco/fix_o365_names.py --ann datasets/objects365/annotations/objects365_minival.json

python3 tools/generate_img_ann_pair.py --json_path datasets/objects365/annotations/objects365_train_fixname.json --image_root datasets/objects365/train/
```

## Expected dataset structure for [OpenImages](https://storage.googleapis.com/openimages/web/download.html#download_manually):
```
$DETECTRON2_DATASETS/
  openimages/
    annotations/
    train/
    validation/
```

## Expected dataset structure for [VisualGenome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html):
```
$DETECTRON2_DATASETS/
  visualgenome/
    VG_100K/
    VG_100K_2/
```

## Expected dataset structure for [SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/):
```
$DETECTRON2_DATASETS/
  SA-1B/
    images/
    sam1b_instance.json
```

## Expected dataset structure for [RefCOCO]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [GQA]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [PhraseCut]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [Flickr30k]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [ODinW]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [SegInW]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [Roboflow100]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [ADE20k]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [ADE-full]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [BDD10k]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [PC459]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [PC59]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [VOC]():
```
$DETECTRON2_DATASETS/
  xxx/
```

## Expected dataset structure for [D3]():
```
$DETECTRON2_DATASETS/
  xxx/
```


