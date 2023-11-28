from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform, TransformList


class AutoAugment(T.Augmentation):
    def __init__(self, cfg):
        super().__init__()
        self.resize = T.AugmentationList(
            [
                T.ResizeShortestEdge(
                    [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                    1333,
                    sample_style="choice",
                ),
            ]
        )
        self.resize_crop_resize = T.AugmentationList(
            [
                T.ResizeShortestEdge([400, 500, 600], 1333, sample_style="choice"),
                T.RandomCrop("absolute_range", (384, 600)),
                T.ResizeShortestEdge(
                    [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                    1333,
                    sample_style="choice",
                ),
            ]
        )

    def __call__(self, aug_input) -> Transform:

        do = self._rand_range(low=0.0, high=1.0)
        if do > 0.5:
            return self.resize(aug_input)
        else:
            return self.resize_crop_resize(aug_input)

    def __repr__(self):
        msgs = [str(self.resize), str(self.resize_crop_resize)]
        return "AutoAugment[{}]".format(", ".join(msgs))
