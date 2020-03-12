import segmentation_models_pytorch as smp
import torch
from losses.BCEJaccard import LossBinary
from catalyst.contrib.criterion import FocalLossMultiClass
from data.final_transforms import hardcore_aug, crazy_custom_aug
from modeller import Model

checkpoint = torch.load("instance.pth")
weights = checkpoint["model_state_dict"]
model = smp.Unet("se_resnext50_32x4d", classes=6)
model.load_state_dict(weights)
model.eval()
model.cuda()
modeler = Model(transforms=hardcore_aug, criterion=FocalLossMultiClass())
modeler.train(model, 10000)
