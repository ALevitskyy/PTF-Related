import cv2
from utils import make_a_cage, overlay_image, overlay_mask
import albumentations

sizes = {"XXS":[3,3],
         "XS":[5,5],
         "S":[10,10],
         "M":[20,20],
         "L":[30,30],
         "XL":[40,40],    
}
offsets = [40*3,18*3] # Dimensions are flipped
refernce_size = [1280, 720]  #DImensions are flipped

class CageMaker():
    def __init__(self, template, sizes, offsets, reference_size):
        self.cages = {key:self.make_cage(template, sizes[key],
                                    offsets, reference_size) for key in sizes}
        self.reference_size = reference_size
    
    def make_cage(self, template, size, offsets, reference_size):
        dim1, dim2, _ = template.shape
        rescale1 = (reference_size[0]/size[0])/(dim1-offsets[0])
        rescale2 = (reference_size[1]/size[1])/(dim2-offsets[1])
        new_template = cv2.resize(template,(int(dim1*rescale1),
                                        int(dim2*rescale2)))
        NCell1 = size[0]+2
        NCell2 = size[1]+2
        offsets = [int(offsets[0]*rescale1),int(offsets[1]*rescale2)]
        result = make_a_cage(new_template, 
                             (NCell1,NCell2), offsets).transpose(1,0,2)
        return result
    
    def get_cage(self,image,size):
        template = self.cages[size]
        template = albumentations.CenterCrop(refernce_size[0],refernce_size[1])\
                             (image = template)["image"]
        return cv2.resize(template,(image.shape[1],image.shape[0]))

class CageOverlay(albumentations.DualTransform):

    def __init__(self, cagemaker, size, always_apply=False, p=1.0):
        super(CageOverlay, self).__init__(always_apply, p)
        self.cagemaker = cagemaker
        self.size = size

    def apply(self, img, **params):
        height,width,_ = img.shape
        self.cropped_overlay = self.cagemaker.get_cage(img,
                                                       self.size)
        self.cropped_overlay=albumentations.ShiftScaleRotate()\
                                   (image = self.cropped_overlay)["image"]
        return overlay_image(img, self.cropped_overlay)
    
    def apply_to_mask(self, img, **params):
        return overlay_mask(img, self.cropped_overlay)