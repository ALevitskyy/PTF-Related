from .utils import targets_aug
from albumentations import (
    Compose,
    RGBShift,
    RandomGamma,
    RandomCrop,
    ShiftScaleRotate,
    VerticalFlip,
    HorizontalFlip,
    RandomRotate90,
    RandomBrightnessContrast,
    GaussNoise,
    PadIfNeeded,
    HueSaturationValue,
    Resize,
    ElasticTransform,
    MedianBlur,
    RandomBrightness,
)
from .augmentations.RandomInfoOverlay import image_dict, RandomInfoOverlay
from .augmentations.CageMaker import default_cage_maker
from .augmentations.RandomCageOverlay import RandomCageOverlay


def transforms(targets):
    aug = targets_aug(
        [
            RandomCrop(256, 256),
            ShiftScaleRotate(),
            HorizontalFlip(),
            VerticalFlip(),
            RandomRotate90(),
            RandomBrightnessContrast(),
            GaussNoise(),
            RGBShift(),
            RandomGamma(),
        ],
        targets,
    )
    return aug


def valid_transforms(targets):
    aug = targets_aug([Resize(512, 1024)], targets)
    return aug


def test_transform(targets):
    original_height, original_width = targets[0].shape[:2]
    # resize_height = (int(original_height/64)+1)*64
    # resize_width = (int(original_width/64)+1)*64
    resize_height = 1280
    resize_width = 2048
    aug = targets_aug(
        [PadIfNeeded(p=1, min_height=resize_height, min_width=resize_width)], targets
    )
    return aug


def very_light_aug(targets):
    aug = targets_aug([RandomCrop(256, 256)], targets)
    return aug


def light_aug(targets):
    aug = targets_aug(
        [
            ShiftScaleRotate(),
            RandomCrop(256, 256),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
        ],
        targets,
    )
    return aug


def hardcore_aug(targets):
    aug = targets_aug(
        [
            ShiftScaleRotate(),
            RandomCrop(256, 256),
            HueSaturationValue(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
            RandomGamma(p=1),
        ],
        targets,
    )
    return aug


def crazy_custom_aug(targets):
    cage_aug = RandomCageOverlay(default_cage_maker, p=0.15)
    info_aug = RandomInfoOverlay(image_dict, max_overlay_num=10, p=0.5)
    aug = targets_aug(
        [
            cage_aug,
            info_aug,
            ShiftScaleRotate(),
            RandomCrop(256, 256),
            ElasticTransform(p=0.3),
            MedianBlur(p=0.3),
            RandomBrightness(p=0.3),
            HueSaturationValue(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
            RandomGamma(p=1),
        ],
        targets,
    )
    return aug
