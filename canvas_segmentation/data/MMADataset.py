import cv2
from torch.utils.data import Dataset
import os
from albumentations import PadIfNeeded
from utils import x_to_torch, y_to_torch, transforms

class MMADataset(Dataset):

    def __init__(self, image_dir, transforms = True, split=130,
                 valid=False, test=False,
                pic_num=5):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir,"targets"))
        if valid==False:
            self.image_list = self.image_list[0:split]
        else:
            self.image_list = self.image_list[split:]
            if test:
                self.image_list=self.image_list[0:pic_num]
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        targets = []
        image_path = os.path.join(
            self.image_dir,
            "inputs",
            self.image_list[idx]
        )
        canvas_path = os.path.join(
            self.image_dir,
            "targets",
            self.image_list[idx]
        )
        image = cv2.imread(image_path)
        image = image[...,::-1]
        canvas_mask = cv2.imread(canvas_path, cv2.IMREAD_UNCHANGED)
        targets.append(image)
        targets.append(canvas_mask)
        if self.test==True:
            return {'features': x_to_torch(PadIfNeeded(p=1,min_height=2048, 
                                                      min_width=2048)(image=image)["image"]),
                'targets': y_to_torch(PadIfNeeded(p=1,min_height=2048, 
                                                      min_width=2048)(image=canvas_mask)["image"])}
        if self.transforms:
            aug = transforms(targets)
            return {'features': x_to_torch(aug['image']),
                'targets': y_to_torch(aug['mask'])}
        else:
            return {'features': x_to_torch(image),
                'targets': y_to_torch(canvas_mask)}