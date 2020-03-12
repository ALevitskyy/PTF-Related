# from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
from copy import deepcopy
import imageio
import os

# true_im = Image.open('true_im.png', 'r')
# color_mask_im = Image.open('color_mask_im.png', 'r')
def get_human_mask(true_im, color_mask_im, path):
    true_im.putalpha(200)
    color_mask_im.putalpha(120)
    blended = Image.alpha_composite(true_im, color_mask_im)
    blended.save(path)
    pass


def get_one_color(input1, color):
    color_array = deepcopy(input1)
    color1 = color[0]
    color2 = color[1]
    color3 = color[2]
    red, green, blue = color_array.T
    select_area = (red == color1) & (blue == color2) & (green == color3)
    color_array[select_area.T] = (255, 255, 255)
    color_array[~select_area.T] = (0, 0, 0)
    return color_array[:, :, 0]


def image_to_BW_mask(color_mask_im, path):
    color_array = np.asarray(color_mask_im.convert("RGB"))
    BW_im = get_one_color(color_array, [255, 0, 0])
    imageio.imwrite(path, BW_im)
    pass


maskdir = "mask_computer"
for subdir, dirs, files in os.walk(maskdir):
    for file in files:
        path = os.path.join(subdir, file)
        path_no_root = os.path.join(*(path.split(os.path.sep)[1:]))
        if not "DS_Store" in path:
            actual_image = Image.open(os.path.join("images", path_no_root), "r")
            mask_image = Image.open(path, "r")
            get_human_mask(
                actual_image, mask_image, os.path.join("human_mask", path_no_root)
            )
            image_to_BW_mask(mask_image, os.path.join("BW_mask", path_no_root))
