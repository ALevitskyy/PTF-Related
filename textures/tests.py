#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:49:02 2019

@author: andriylevitskyy
"""
import numpy as np
import cv2
from TextureConverter import DensePoseMethods
import matplotlib.pyplot as plt
def atlas2texturetest():
    Tex_Atlas = cv2.imread('texture_from_SURREAL.png')[:,:,::-1]/255.
    TextureIm  = np.zeros([24,200,200,3]);
    for i in range(4):
        for j in range(6):
            TextureIm[(6*i+j) , :,:,:] = Tex_Atlas[ 
                                            (200*j):(200*j+200) , 
                                            (200*i):(200*i+200) ,: ]
    methods = DensePoseMethods()
    new_image = methods.atlas_to_texture(TextureIm)
    return new_image
texture = atlas2texturetest()
plt.imshow(texture)
cv2.imwrite("new_texture.png",texture[:,:,::-1]*255)
