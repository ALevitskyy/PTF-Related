# import sys
# sys.path.append("/usr/local/lib/python3.7/site-packages")
import albumentations
import cv2
import numpy as np
from copy import deepcopy
from PIL import Image
import random
from config import template, sizes


def load_cv2_RGB(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def add_horizontal_cell_to_cage(array_cage, array_cell, offset=23 * 3):
    # offset = 23 by experimentations
    _, cage_width, _ = array_cage.shape
    _, cell_width, _ = array_cell.shape
    image1 = np.hstack([array_cage, array_cell])
    image1[:, cage_width:, :] = 0
    image1 = image1[:, 0 : (cage_width + cell_width - offset), :]
    image2 = np.hstack([array_cage, array_cell])
    image2[:, 0:cage_width, :] = 0
    image2 = image2[:, offset:, :]
    return np.array(
        Image.alpha_composite(Image.fromarray(image2), Image.fromarray(image1))
    )


def add_vertical_cell_to_cage(array_cage, array_cell, offset=40 * 3):
    # offset = 40 by experimentations, need to work more
    cage_height, _, _ = array_cage.shape
    cell_height, _, _ = array_cell.shape
    image1 = np.vstack([array_cage, array_cage])
    image1[cage_height:, :, :] = 0
    image1 = image1[0 : (cage_height + cell_height - offset), :, :]
    image2 = np.vstack([array_cage, array_cell])
    image2[0:cage_height, :, :] = 0
    image2 = image2[offset:, :, :]
    return np.array(
        Image.alpha_composite(Image.fromarray(image2), Image.fromarray(image1))
    )


def make_a_cage(basecell, dimensions, offsets=[23 * 3, 40 * 3]):
    cage = basecell
    for col in range(1, dimensions[1]):
        cage = add_vertical_cell_to_cage(cage, np.flip(basecell, axis=0), offsets[1])
    cage_unit = cage
    for row in range(1, dimensions[0]):
        cage = add_horizontal_cell_to_cage(cage, cage_unit, offsets[0])
    return cage


class CageMaker:
    def __init__(self, template_dict, sizes):
        self.template_dict = template_dict
        self.cages = {}
        for template in template_dict:
            template_image = cv2.imread(template, cv2.IMREAD_UNCHANGED)
            example_cage = {
                key: self.make_cage(
                    template_image,
                    sizes[key],
                    template_dict[template][0],
                    template_dict[template][1],
                )
                for key in sizes
            }
            self.cages[template] = example_cage

    def make_cage(self, template, size, offsets, reference_size):
        dim1, dim2, _ = template.shape
        rescale1 = (reference_size[0] / size[0]) / (dim1 - offsets[0])
        rescale2 = (reference_size[1] / size[1]) / (dim2 - offsets[1])
        new_template = cv2.resize(
            template,
            (int(dim1 * rescale1), int(dim2 * rescale2)),
            interpolation=cv2.INTER_AREA,
        )
        NCell1 = size[0] + 2
        NCell2 = size[1] + 2
        offsets = [int(offsets[0] * rescale1), int(offsets[1] * rescale2)]
        result = make_a_cage(new_template, (NCell1, NCell2), offsets).transpose(1, 0, 2)
        return result

    def get_cage(self, image, size):
        # Get random cage))
        index = random.choice(list(self.cages))
        cages = self.cages[index]
        reference_size = self.template_dict[index][1]
        template = cages[size]
        # The 2 lines below are very stupid
        template = albumentations.CenterCrop(reference_size[0], reference_size[1])(
            image=template
        )["image"]
        image = cv2.resize(
            template, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
        )
        return image


default_cage_maker = CageMaker(template, sizes)
