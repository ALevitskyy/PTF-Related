from inference import Infer
from models.AlbuNet.AlbuNet import AlbuNet
import segmentation_models_pytorch as smp
inferer = Infer(threshold=0.75)
model = smp.Unet("se_resnext50_32x4d",
        classes=6)
model.cuda()
inferer.inference(model)
