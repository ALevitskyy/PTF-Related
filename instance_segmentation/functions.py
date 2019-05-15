import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict

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

def plot_masks(color_mask):
    for i in color_codes:
        color_code=np.array(color_codes[i])
        color_code= color_code[::-1]
        mini,maxi = get_min_max(color_code)
        print(mini,maxi)
        mask = cv2.inRange(color_mask, mini,maxi)/255
        print(i)
        plt.imshow(mask)
        plt.show()

def get_masks(color_mask):
    maski = []
    for i in color_codes:
        color_code=np.array(color_codes[i])
        color_code= color_code[::-1]
        mini,maxi = get_min_max(color_code)
        mask = cv2.inRange(color_mask, mini,maxi)
        maski.append(mask)
    return maski

def get_contour(mask,width,image,color):
    im2, contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    image_copy = deepcopy(image)
    contours2 = cv2.drawContours(image_copy,contours, -1, color, width)
    return contours2

def plot_contours(color_mask, image):
    for mask in get_masks(color_mask):
        image2 = get_contour(mask,10,image,(0,255,0))
        plt.imshow(image2)
        plt.show()
        
def color2targets(color_mask):
    targets = get_masks(color_mask)
    zero = np.zeros(a[0].shape)
    contour1 = get_contour(a[0],10,zero,(255))
    zero = np.zeros(a[0].shape)
    contour2 = get_contour(a[1],10,zero,(255))
    merged = np.logical_or(contour1>125,contour2>125)*255
    targets.append(merged)
    return np.array(targets)/255
