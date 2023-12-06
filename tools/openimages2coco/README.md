# openimages2coco
Convert [Open Images](https://storage.googleapis.com/openimages/web/index.html "Open Images Homepage") annotations into [MS Coco](http://cocodataset.org "MS Coco Homepage") format to make it a drop in replacement.

### Functionality

-  `convert_annotations.py` will load the original .csv annotation files from Open Images, convert the annotations into the list/dict based format of [MS Coco annotations](http://cocodataset.org/#format-data) and store them as a .json file in the same folder.

- `convert_predictions.py` loads a .json file with predictions in the coco format and save them as .csv in the OpenImages prediction fromat at the same location.

### Installation

Download the CocoAPI from https://github.com/cocodataset/cocoapi \
Install Coco API:
```
cd PATH_TO_COCOAPI/PythonAPI
make install
```

Download Open Images from https://storage.googleapis.com/openimages/web/download.html \
-> Store the images in three folders called: `train, val and test` \
-> Store the annotations for all three splits in a separate folder called: `annotations`

### Converting Annotations

Run conversion of bounding box annotations:
```
python3 convert_annotations.py -p PATH_TO_OPENIMAGES --task bbox
```

The convert instance masks to the [Coco panoptic format](http://cocodataset.org/#panoptic-2019).
The masks have to be placed in `annotations/SPLIT_masks/`
```
python3 convert_annotations.py -p PATH_TO_OPENIMAGES --task panoptic
```

Currently adding the instance masks to the annotations as done for coco is not supported becasue the resulting `json` file would be extremely large.


### Converting Predictions

Run conversion of bounding box predictions:
```
python3 convert_predictions.py -p PATH_TO_PREDICTIONS --subset validation
```

The subset is necessary to get the correct image sizes. Alternatively they can be inferred directly from the images:
```
python3 convert_predictions.py -p PATH_TO_PREDICTIONS --image_dir PATH_TO_IMAGES
```

Currently only bounding box predictions are supported.


### Dataset Versions

The toolkit supports multiple versions of the dataset including `v4`, `v5`, `v6` and `challenge_2019`.
For example the `bbox` annotations of `challenge_2019` can be converted like:
```
python3 convert_annotations.py -p PATH_TO_OPENIMAGES --version challenge_2019 --task bbox
```
For panoptica nnotations the masks have to be placed in `annotations/challenge_2019_$SPLIT_masks/` before running:
```
python3 convert_annotations.py -p PATH_TO_OPENIMAGES --version challenge_2019 --task panoptic
```

Note, that different annotation files have to be downloaded to `annotations` for this purpose.
The files for the `challenge_2019` set can be found here: https://storage.googleapis.com/openimages/web/challenge2019_downloads.html


### Using Converted Annotations

The generated annotations can be loaded and used with the standard MS Coco tools:
```
from pycocotools.coco import COCO

# Example for the validation set
openimages = COCO('PATH_TO_OPENIMAGES/annotations/openimages_v6_val_bbox.json')
```