from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform, TransformList


class LargeScaleJitter(T.Augmentation):
    def __init__(self, cfg):
        super().__init__()

        image_size = cfg.INPUT.LSJ.IMAGE_SIZE
        min_scale = cfg.INPUT.LSJ.MIN_SCALE
        max_scale = cfg.INPUT.LSJ.MAX_SCALE
        # pad_value = 128.0
        pad_value = 1.0 * sum(cfg.MODEL.PIXEL_MEAN) / len(cfg.MODEL.PIXEL_MEAN)
        seg_pad_value = cfg.INPUT.SEG_PAD_VALUE

        self.resize_crop = T.AugmentationList(
            [
                T.ResizeScale(
                    min_scale=min_scale,
                    max_scale=max_scale,
                    target_height=image_size,
                    target_width=image_size,
                ),
                T.FixedSizeCrop(
                    crop_size=(image_size, image_size),
                    pad_value=pad_value,
                    seg_pad_value=seg_pad_value,
                ),
            ]
        )

    def __call__(self, aug_input) -> Transform:

        return self.resize_crop(aug_input)

    def __repr__(self):
        msgs = str(self.resize_crop)
        return "LargeScaleJitter[{}]".format(msgs)
