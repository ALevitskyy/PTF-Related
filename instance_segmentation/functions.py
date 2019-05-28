#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 01:11:57 2019

@author: andriylevitskyy
"""


import numpy as np
import cv2
from copy import deepcopy
from collections import OrderedDict

#Photoshop Mask ==> stack of individual masks

color_codes = OrderedDict()
color_codes["fighter1"]=[0,0,255]
color_codes["fighter2"]=[0,255,0]
color_codes["canvas"]=[255,0,0]
color_codes["ref"]=[255,0,255]
color_codes["background"]=[0,255,255]
color_codes["overlay"]=[255,255,0]

def get_min_max(color):
    mini = []
    maxi = []
    for i in color:
        if i<50:
            mini.append(0)
            maxi.append(50)
        else:
            mini.append(200)
            maxi.append(255)
    return np.array(mini), np.array(maxi)

def get_masks(color_mask):
    maski = []
    for i in color_codes:
        color_code=np.array(color_codes[i])
        color_code= color_code[::-1]
        mini,maxi = get_min_max(color_code)
        mask = cv2.inRange(color_mask, mini,maxi)
        maski.append(mask)
    return maski

# Individual masks stack ==> Segmentations masks stack

def outlines(img):
    return cv2.morphologyEx(img,
                     cv2.MORPH_GRADIENT,
                     np.ones((5,5),np.uint8))

def merge_human(mask1, mask2):
    combo = mask1+mask2
    smoothed = cv2.morphologyEx(combo,
                         cv2.MORPH_CLOSE,
                        np.ones((2,2),np.uint8))
    return smoothed

def get_internal_bound(mask1, mask2):
    bound0 = outlines(mask1)
    bound1 = outlines(mask2)
    internal_bounds = np.logical_and(bound1>0, bound0>0)
    return internal_bounds

def masks2targets(maski):
    humans = merge_human(maski[0], maski[1])
    internal_bound = get_internal_bound(maski[0], maski[1])
    targets = deepcopy(maski)
    targets[0] = humans
    targets[1] = internal_bound*255
    return np.array(targets,np.uint8)

# Human + Boundaries ==> Components

def merge_boundaries(targets):
    humans = deepcopy(targets[0])
    humans = humans/255
    humans[targets[1]>120]=0.5
    return humans

def get_components(merged,targets):
    n_comp,labels = cv2.connectedComponents(
                        merged.astype(np.uint8))
    smooth_labels = cv2.morphologyEx(
                         labels.astype(np.uint8),
                         cv2.MORPH_DILATE,
                         np.ones((7,7),np.uint8))
    smooth_labels[targets[0]==0]=0
    return smooth_labels

# Components ==> 2 People (to be finished)
    
def is_neighbor(comp1,comp2):
    comp1_dil = cv2.morphologyEx(comp1.astype(np.uint8),
                         cv2.MORPH_DILATE,
                        np.ones((2,2),np.uint8))
    if np.sum(np.logical_and(comp1_dil==1,comp2==1))>0:
        return True
    else:
        return False

def get_box(patch):
    coords = np.where(patch)
    minim = (min(coords[0]),min(coords[1]))
    maxim = (max(coords[0]),max(coords[1]))
    box = np.array([[minim[0],minim[1]],
                [minim[0],maxim[1]],
                [maxim[0],minim[1]],
                [maxim[0],maxim[1]],
                ])
    return box

def minim_dist(box1,box2):
    concat = []
    for i in range(4):
        concat.append(np.sum((box1[i,:]-box2)**2,axis=1))
    concat = np.array(concat)
    return np.min(concat)