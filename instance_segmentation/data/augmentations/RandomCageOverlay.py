# import sys
# sys.path.append("/usr/local/lib/python3.7/site-packages")
import cv2
import numpy as np
from copy import deepcopy
from .CageMaker import default_cage_maker
from albumentations.core.transforms_interface import DualTransform
import random
import albumentations


def overlay_mask(mask_bottom, image_top):
    height, width = mask_bottom.shape
    overlay = cv2.resize(image_top, (width, height))
    result = deepcopy(mask_bottom)
    alpha_mask = overlay[:, :, 3] != 0
    result[alpha_mask] = 0
    return result


def overlay_image(image_bottom, image_top):
    height, width, channels = image_bottom.shape
    overlay = cv2.resize(image_top, (width, height))
    result = deepcopy(image_bottom)
    alpha_mask = np.broadcast_to(
        np.reshape(overlay[:, :, 3] != 0, (height, width, 1)), result.shape
    )
    # print(alpha_mask.shape,result.shape, overlay.shape)
    result[alpha_mask] = overlay[:, :, :3][alpha_mask]
    return result


class RandomCageOverlay(DualTransform):
    """
        Try something very simple first. Overlays one image above the other
        """

    def __init__(self, cagemaker, sizes=["XXS", "XS", "S"], always_apply=False, p=1.0):
        super(RandomCageOverlay, self).__init__(always_apply, p)
        self.cagemaker = cagemaker
        self.sizes = sizes

    def __call__(self, force_apply=False, **kwargs):
        size = random.choice(self.sizes)
        self.cropped_overlay = self.cagemaker.get_cage(kwargs["image"], size)
        self.cropped_overlay = albumentations.ShiftScaleRotate(
            shift_limit=0.3, scale_limit=0.5, rotate_limit=360
        )(image=self.cropped_overlay)["image"]
        # More albumentations!!!
        # self.cropped_overlay = albumentations.\
        #    ShiftScaleRotate(,p=1)(image = self.cropped_overlay)["image"]
        return super(RandomCageOverlay, self).__call__(
            force_apply=force_apply, **kwargs
        )

    def apply(self, img, **params):
        return overlay_image(img, self.cropped_overlay)

    def apply_to_mask(self, img, **params):
        return overlay_mask(img, self.cropped_overlay)


default_cage_overlay = RandomCageOverlay(default_cage_maker)
