import cv2
from torch.utils.data import Dataset
import os
from albumentations import PadIfNeeded
from .utils import x_to_torch, y_to_torch
from glob import glob
import numpy as np
class InferenceDataset(Dataset):

    def __init__(self, image_folder,
                 transforms = None,
                 num_batches=1,# to be able to split inferences
                 batch_id=0):
        self.image_list = glob(image_folder+"/*.png",recursive = True)+\
                                   glob(image_folder+"/*.jpg",recursive = True)
        self.image_list = [[i] for i in self.image_list] # to be similar with MMADatset
        length = len(self.image_list)
        batch_size= int(length/num_batches)+1
        self.image_list = self.image_list[(batch_id*batch_size):((batch_id+1)*batch_size)]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        targets = []
        image_path = self.image_list[idx][0]
        image = cv2.imread(image_path)
        image = image[...,::-1] #BGR==>RGB
        height,width = image.shape[:2]
        canvas_mask = np.zeros((height,width,1))
        targets.append(image)
        targets.append(canvas_mask)
        if self.transforms is not None:
            aug = self.transforms(targets)
        else:
            aug = {"image":image,
                   "mask":canvas_mask}
        return {"features": x_to_torch(aug["image"])}


