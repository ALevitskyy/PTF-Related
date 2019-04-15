import cv2
import numpy as np
from copy import deepcopy
from PIL import Image

def load_cv2_RGB(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def overlay_mask(mask_bottom,image_top):
    height, width, channels = mask_bottom.shape
    overlay = cv2.resize(image_top,(width,height))
    result = deepcopy(mask_bottom)
    alpha_mask = overlay[:,:,3]!=0
    result[alpha_mask] = 0
    return result

def overlay_image(image_bottom,image_top):
    height, width, channels = image_bottom.shape
    overlay = cv2.resize(image_top,(width,height))
    result = deepcopy(image_bottom)
    alpha_mask = np.broadcast_to(np.reshape(overlay[:,:,3]!=0,
                                            (height,width,1)),
                                        result.shape)
    print(alpha_mask.shape,result.shape, overlay.shape)
    result[alpha_mask] = overlay[:,:,:3][alpha_mask]
    return result

def add_horizontal_cell_to_cage(array_cage,array_cell,offset=23*3):
    # offset = 23 by experimentations
    _,cage_width,_ = array_cage.shape
    _,cell_width,_ = array_cell.shape
    image1= np.hstack([array_cage,array_cell])
    image1[:,cage_width:,:]=0
    image1=image1[:,0:(cage_width+cell_width-offset),:]
    image2= np.hstack([array_cage,array_cell])
    image2[:,0:cage_width,:]=0
    image2=image2[:,offset:,:]
    return np.array(Image.alpha_composite(
        Image.fromarray(image2),
        Image.fromarray(image1)
    ))
    
def add_vertical_cell_to_cage(array_cage,array_cell,offset=40*3):
    # offset = 40 by experimentations, need to work more
    cage_height,_,_ = array_cage.shape
    cell_height,_,_ = array_cell.shape
    image1= np.vstack([array_cage,array_cage])
    image1[cage_height:,:,:]=0
    image1=image1[0:(cage_height+cell_height-offset),:,:]
    image2= np.vstack([array_cage,array_cell])
    image2[0:cage_height,:,:]=0
    image2=image2[offset:,:,:]
    return np.array(Image.alpha_composite(
        Image.fromarray(image2),
        Image.fromarray(image1)
    ))
  
def make_a_cage(basecell,dimensions, offsets=[23*3,40*3]):
    cage = basecell
    for col in range(1,dimensions[1]):
        cage = add_vertical_cell_to_cage(
                cage,
                np.flip(basecell,axis = 0),
                offsets[1])
    cage_unit=cage
    for row in range(1,dimensions[0]):
        cage = add_horizontal_cell_to_cage(
                cage,
                cage_unit,
                offsets[0]
                )
    return cage