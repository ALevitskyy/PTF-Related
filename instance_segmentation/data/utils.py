import numpy as np
import torch
from albumentations import Compose
from torchvision.transforms import Normalize

def x_to_torch(x):
    unnormalized = torch.from_numpy(np.moveaxis(x.astype(np.float32) / 255., -1, 0)) #moveaxis to move channels
    return Normalize(mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225))(unnormalized)

def y_to_torch(y):
    y_converted = np.argmax(y,axis=0).astype(np.float32)
    return torch.from_numpy(y_converted)

def targets_aug(transforms, image, targets):
    return Compose(transforms, p=1)(image=image,masks=targets)

