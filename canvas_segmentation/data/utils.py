import numpy as np
import torch
from albumentations import Compose, RGBShift, RandomGamma,\
        RandomCrop, ShiftScaleRotate, VerticalFlip, HorizontalFlip,\
         RandomRotate90, RandomBrightnessContrast, GaussNoise
from torchvision.transforms import Normalize

def x_to_torch(x):
    unnormalized = torch.from_numpy(np.moveaxis(x.astype(np.float32) / 255., -1, 0))
    return Normalize(mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225))(unnormalized)
    
def y_to_torch(y):
    return torch.from_numpy(np.expand_dims(y.astype(np.float32) / 255., axis=0))

def open_fn(x):
    return {'features': x['image'], 'targets': x['mask']}

def targets_aug(transforms, targets):
    target = {}
    for i, mask in enumerate(targets[1:]):
        target['mask' + str(i)] = 'mask'
    return Compose(transforms, p=1, additional_targets=target)(image=targets[0],
                                                                mask=targets[1])
  
def transforms(targets):
    aug = targets_aug([RandomCrop(256, 256),
        ShiftScaleRotate(),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        RandomBrightnessContrast(),
        GaussNoise(), RGBShift(), RandomGamma()
    ], targets)
    return aug