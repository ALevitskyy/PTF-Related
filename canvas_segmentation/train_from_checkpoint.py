 import segmentation_models_pytorch as smp
 import torch
 from losses.BCEJaccard import LossBinary
 from data.final_transforms import hardcore_aug
 from modeller import Model
 checkpoint = torch.load("last.pth")
 weights = checkpoint["model_state_dict"]
 model = smp.Unet("se_resnext50_32x4d")
 model.load_state_dict(weights)
 model.eval()
 model.cuda()
 modeler = Model(
           transforms = hardcore_aug,
           criterion = LossBinary(0.3)
           )
 modeler.train(model,10000)
