from data.utils import y_to_torch
import cv2
from data.processing_functions import color2targets
from catalyst.contrib.criterion import FocalLossMultiClass
test_path = "./images/razmetka/razmetka1/first_batch/masks/20.png"
test_image = cv2.imread(test_path)
print(test_image.shape)
targets = color2targets(test_image)
print(targets.shape)
torchok = y_to_torch(targets)
print(torchok.shape)
print(torchok.long().shape)
print(FocalLossMultiClass(torchok, torchok))
