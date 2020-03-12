#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:49:02 2019

@author: andriylevitskyy
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Combines 3 highest detection from DensePose over whole video into 2 textures
def iuv2atlas(frame, iuv, inds):
    # Converts outputs of densepose into 3 textures
    inds[np.logical_and(inds == 0, iuv.any(axis=-1))] = 3
    atlas = np.zeros((3, 24, 256, 256, 3))
    for detect_index in range(1, 4):
        frame_detected = frame[inds == detect_index]
        iuv_detected = iuv[inds == detect_index]
        I, U, V = iuv_detected.transpose()
        atlas[detect_index - 1, I - 1, 255 - V, U, :] = frame_detected
    return atlas.astype(int)


def combine_atlas(atlas):
    # Required for visualization only
    combined = np.zeros((256 * 4, 256 * 6, 3))
    for row in range(4):
        for col in range(6):
            combined[
                row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256, :
            ] = atlas[row * 6 + col, :, :, :]
    return combined


def visualize_atlas(atlas):
    # Visualize collected atlases
    combo = [combine_atlas(atlas[0]), combine_atlas(atlas[1])]
    for i in combo:
        plt.imshow(i[:, :, ::-1])
        plt.show()
    pass


def index_atlas(indx):
    # Loads outputs of DensePose
    frame = cv2.imread("video/frames/" + str(indx) + ".jpg")
    iuv = cv2.imread("infer_out/" + str(indx) + "_IUV.png")
    inds = cv2.imread("infer_out/" + str(indx) + "_INDS.png", 0)
    return iuv2atlas(frame, iuv, inds)


def mse2atlasses(atlas1, atlas2):
    # Metric to calculate KNN on
    both_mask = np.logical_and(atlas1.any(axis=-1), atlas2.any(axis=-1))
    diff = ((atlas1 / 255 - atlas2 / 255) ** 2)[both_mask].mean(axis=-1)
    return np.mean(diff)


def get_brother_atlas(atlas, candidates, threshold):
    # Two simple, results in 2 similar textres
    # Need to find minimum, then check if greater than threshold
    for candidate in candidates:
        mse = mse2atlasses(atlas, candidate)
        print(mse)
        if mse < threshold:
            return candidate
    return np.zeros(atlas.shape)


# Part responsible for actual running of algorithm
threshold = 0.03
init_atlas = index_atlas(0)[:2]
for i in range(1, 7000, 20):  # sample every 20 frames
    print(i)
    new_atlas = index_atlas(i)
    new_atlas1 = get_brother_atlas(init_atlas[0], new_atlas, threshold)
    new_atlas2 = get_brother_atlas(
        init_atlas[1], new_atlas, threshold
    )  # get 2 nearest textures
    new_atlas = np.array([new_atlas1, new_atlas2])
    new_mask = new_atlas.any(axis=-1)
    init_mask = init_atlas.any(axis=-1)
    mask_both = np.logical_and(new_mask, init_mask)
    init_atlas[~init_mask] = new_atlas[~init_mask]
    init_atlas[mask_both] = 0.8 * init_atlas[mask_both] + 0.2 * new_atlas[mask_both]
    # visualize_atlas(new_atlas)
    # visualize_atlas(init_atlas)
cv2.imwrite("blablabla.png", combine_atlas(init_atlas[1]).transpose(1, 0, 2))
