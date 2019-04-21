import numpy as np
import torch
from albumentations import Compose
from torchvision.transforms import Normalize

def x_to_torch(x):
    unnormalized = torch.from_numpy(np.moveaxis(x.astype(np.float32) / 255., -1, 0)) #moveaxis to move channels
    return Normalize(mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225))(unnormalized)

def y_to_torch(y):
    return torch.from_numpy(np.expand_dims(y.astype(np.float32) / 255., axis=0))

def targets_aug(transforms, targets):
    return Compose(transforms, p=1)(image=targets[0],mask=targets[1])

