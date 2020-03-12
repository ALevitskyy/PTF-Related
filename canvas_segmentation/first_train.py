from models.AlbuNet.AlbuNet import AlbuNet
from modeller import Model
from data.final_transforms import very_light_aug, light_aug, hardcore_aug
from losses.BCEJaccard import LossBinary
import segmentation_models_pytorch as smp

model1_stage = Model(transforms=very_light_aug,)
model2_stage = Model(transforms=light_aug, criterion=LossBinary(0.3))
model3_stage = Model(transforms=hardcore_aug, criterion=LossBinary(0.3))
# net = AlbuNet()
net = smp.Unet("se_resnext50_32x4d", encoder_weights="imagenet")
net.cuda()
print("No Augmentations")
model1_stage.train(net, 30)
print("Light Augmentations")
model2_stage.train(net, 40)
print("Hardcore Augmentations")
model3_stage.train(net, 50)
