import cv2
from torch.utils.data import Dataset
import os
from albumentations import PadIfNeeded
from .utils import x_to_torch, y_to_torch
from .processing_functions import color2targets
import numpy as np


class MMADataset(Dataset):
    def __init__(self, split_file, transforms=None, valid=False):
        with open(split_file, "r") as f:
            image_list = eval(f.read())
            if valid:
                self.image_list = image_list["valid"]
            else:
                self.image_list = (
                    image_list["train"] * 4
                )  # run validation every 5th epoch
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        targets = []
        image_path = self.image_list[idx][0]
        # print(image_path)
        canvas_path = self.image_list[idx][1]
        image = cv2.imread(image_path)
        image = image[..., ::-1]  # BGR==>RGB
        color_mask = cv2.imread(canvas_path)
        targets = color2targets(color_mask)
        if self.transforms is not None:
            aug = self.transforms(image, targets)
        else:
            aug = {"image": image, "masks": targets}
        return {
            "features": x_to_torch(aug["image"]),
            "targets": y_to_torch(np.array(aug["masks"])),
        }
