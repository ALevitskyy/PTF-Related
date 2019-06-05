#import sys
#sys.path.append("/usr/local/lib/python3.7/site-packages")
import cv2
from copy import deepcopy 
import numpy as np
from albumentations.core.transforms_interface import DualTransform
import albumentations
from random import randint
import random
import pickle
from config import image_dict

def overlay_mask(mask_bottom, image_top, coordinates):
    height, width, channels = image_top.shape
    result = deepcopy(mask_bottom)
    result[coordinates[0]:(coordinates[0]+height),
           coordinates[1]:(coordinates[1]+width)]\
           [image_top[:,:,3]!=0]=0
    return result

def overlay_image(image_bottom, image_top, coordinates):
    height, width, channels = image_top.shape
    result = deepcopy(image_bottom)
    alpha_mask = np.broadcast_to(np.reshape(image_top[:,:,3]!=0,
                                            (height,width,1)),
                                 (height,width,3))
    result[coordinates[0]:(coordinates[0]+height),
           coordinates[1]:(coordinates[1]+width),:]\
                [alpha_mask] = image_top[:,:,:3][alpha_mask]
    return result
	


class RandomInfoOverlay(DualTransform):
    """
        Try something very simple first. Overlays one image above the other
        """
    
    def __init__(self, overlay_dict, max_overlay_num = 5, 
                 always_apply=False, p=1.0):
        super(RandomInfoOverlay,self).__init__(always_apply, p)
        self.overlay_dict = overlay_dict
        self.overlays = {i:cv2.imread(i,cv2.IMREAD_UNCHANGED)\
                         for i in overlay_dict}
        self.max_overlay_num = max_overlay_num
    def __call__(self, force_apply=False, **kwargs):
        
        self.mask = kwargs["mask"]
        self.image = kwargs["image"]
        im_height, im_width, _ = kwargs["image"].shape
        overlay_num = random.randint(1,self.max_overlay_num)
        for i in range(overlay_num):
            index = random.choice(list(self.overlay_dict))
            ref_overlay_height, ref_overlay_width = self.overlay_dict[index]
            overlay_height, overlay_width, _ = self.overlays[index].shape
            overlay = deepcopy(self.overlays[index])
            overlay = albumentations.Resize(
                  int(im_height/ref_overlay_height*overlay_height),
                  int(im_width/ref_overlay_width*overlay_width), p = 1)\
            (image = overlay)["image"] # Resize to target image dimensions
            #More albumentations!!!
            #overlay = albumentations.ShiftScaleRotate(shift_limit=0.01,
            #                                          scale_limit=0.5,
            #                                          rotate_limit=90, p = 1)\
            #                                (image = overlay)["image"]
            #overlay = albumentations.Blur(p = 0.5)(image = overlay)["image"]
            #overlay = albumentations.ChannelShuffle(p = 0.5)(image = overlay)["image"]
            overlay_height, overlay_width, _ = overlay.shape
            coordinate1 = random.randint(0,im_height - overlay_height - 1)
            coordinate2 = random.randint(0,im_width - overlay_width - 1)
            self.mask = overlay_mask(self.mask, overlay,
                                     [coordinate1, coordinate2])
            self.image = overlay_image(self.image, overlay,
                                       [coordinate1, coordinate2])
            
        return super(RandomInfoOverlay,self).__call__(force_apply=force_apply, **kwargs)
    
    def apply(self, img, **params):
        return self.image
    
    def apply_to_mask(self, img, **params):
        return self.mask

default_info_overlay = RandomInfoOverlay(image_dict, max_overlay_num = 10)
